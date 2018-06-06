/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <unordered_set>
#include <iostream>
#include "./imperative_utils.h"
#include "./cached_op.h"
#include "../executor/exec_pass.h"
#include "../profiler/profiler.h"


namespace mxnet {
namespace imperative {

DMLC_REGISTER_PARAMETER(CachedOpConfig);

struct CachedOpState {
  CachedOpState(const Context& context_,
                const nnvm::Graph& fwd_graph_,
                const nnvm::Graph& full_graph_) {
    context = context_;
    runtime.fwd_graph = fwd_graph_;
    runtime.full_graph = full_graph_;

    size_t max_nodes = runtime.full_graph.indexed_graph().num_nodes();
    runtime.fwd_graph.attrs["context"] = std::make_shared<dmlc::any>(
        std::vector<Context>(runtime.fwd_graph.indexed_graph().num_nodes(), context));
    runtime.full_graph.attrs["context"] = std::make_shared<dmlc::any>(
        std::vector<Context>(max_nodes, context));
  }

  std::mutex mutex;
  Context context;
  imperative::GraphRuntime runtime;
  std::multimap<size_t, NDArray> fwd_reuse_pool;
  std::multimap<size_t, NDArray> bwd_reuse_pool;
};

CachedOp::CachedOp(
    const nnvm::Symbol& sym,
    const std::vector<std::pair<std::string, std::string> >& flags) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const auto _copy = Op::Get("_copy");

  config_.Init(flags);

  // construct forward graph
  {
    NodeEntryMap<int> dedup_out;
    for (const auto& i : sym.outputs) {
      if (dedup_out.count(i)) {
        NodePtr copy_node = Node::Create();
        copy_node->attrs.op = _copy;
        copy_node->attrs.name =
            i.node->attrs.name + "_copy" + std::to_string(dedup_out[i]++);
        copy_node->inputs.emplace_back(i);
        if (_copy->attr_parser != nullptr) {
          _copy->attr_parser(&(copy_node->attrs));
        }
        fwd_graph_.outputs.push_back(NodeEntry{copy_node, 0, 0});
      } else {
        dedup_out.insert({i, 0});
        fwd_graph_.outputs.push_back(i);
      }
    }
    const auto& idx = fwd_graph_.indexed_graph();
    CHECK_GE(idx.input_nodes().size(), 1) << "CachedOp requires at least 1 input";
    num_forward_nodes_ = idx.num_nodes();
    num_forward_entries_ = idx.num_node_entries();
    num_forward_inputs_ = idx.input_nodes().size();
    num_forward_outputs_ = idx.outputs().size();

    std::vector<uint32_t> ref_count(num_forward_entries_, 0);
    for (const auto& i : idx.input_nodes()) ++ref_count[idx.entry_id(i, 0)];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    for (size_t i = 0; i < num_forward_nodes_; ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }

    fwd_graph_.attrs["forward_ref_count"] =
        std::make_shared<dmlc::any>(std::move(ref_count));

    inlining_ = !config_.static_alloc &&
        (num_forward_nodes_ - num_forward_inputs_) <= config_.inline_limit;
  }

  // Set params
  {
    if (config_.data_indices.ndim() || config_.param_indices.ndim()) {
      CHECK_EQ(config_.data_indices.ndim() + config_.param_indices.ndim(),
               num_forward_inputs_);
    } else {
      std::vector<uint32_t> tmp;
      for (size_t i = 0; i < num_forward_inputs_; ++i) {
        tmp.push_back(i);
      }
      config_.data_indices.assign(tmp.begin(), tmp.end());
    }
  }

  // construct backward graph
  {
    ograd_entries_.reserve(num_forward_outputs_);
    for (size_t i = 0; i < num_forward_outputs_; ++i) {
      ograd_entries_.emplace_back(NodeEntry{Node::Create(), 0, 0});
    }

    std::vector<NodeEntry> xs;
    const auto& idx = fwd_graph_.indexed_graph();
    for (size_t i = 0; i < num_forward_inputs_; ++i) {
      auto nid = idx.input_nodes()[i];
      if (idx.mutable_input_nodes().count(nid)) continue;
      fwd_input_to_grad_output_[i] = xs.size();
      xs.emplace_back(NodeEntry{idx[nid].weak_ref.lock(), 0, 0});
    }

    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";

    grad_graph_ = pass::Gradient(
        fwd_graph_, fwd_graph_.outputs, xs, ograd_entries_,
        exec::AggregateGradient, nullptr, nullptr,
        zero_ops, "_copy");
  }

  // construct full graph
  {
    full_graph_.outputs = fwd_graph_.outputs;
    bwd_output_reqs_ = std::vector<OpReqType>(grad_graph_.outputs.size(), kWriteTo);
    for (const auto& i : grad_graph_.outputs) full_graph_.outputs.emplace_back(i);
    const auto& idx = full_graph_.indexed_graph();

    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes_; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) {
         ++ref_count[idx.entry_id(j)];
      }
    }

    auto full_ref_count = fwd_graph_.GetAttr<std::vector<uint32_t> >("forward_ref_count");
    for (size_t i = 0; i < num_forward_entries_; ++i) full_ref_count[i] += ref_count[i];
    fwd_graph_.attrs["full_ref_count"] =
        std::make_shared<dmlc::any>(std::move(full_ref_count));

    for (uint32_t i = 0; i < ograd_entries_.size(); ++i) {
      if (!idx.exist(ograd_entries_[i].node.get())) continue;
      auto eid = idx.entry_id(ograd_entries_[i]);
      if (ref_count[eid] > 0) {
        bwd_ograd_dep_.push_back(i);
      }
    }
    save_inputs_.resize(num_forward_inputs_, false);
    for (uint32_t i = 0; i < num_forward_inputs_; ++i) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      if (ref_count[eid] > 0) {
        save_inputs_[i] = true;
        bwd_in_dep_.push_back(i);
      }
    }
    save_outputs_.resize(idx.outputs().size(), false);
    for (uint32_t i = 0; i < num_forward_outputs_; ++i) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      if (ref_count[eid] > 0) {
        save_outputs_[i] = true;
        bwd_out_dep_.push_back(i);
      }
    }
  }
}

CachedOp::~CachedOp() {
}

std::vector<nnvm::NodeEntry> CachedOp::Gradient(
    const nnvm::NodePtr& node,
    const std::vector<nnvm::NodeEntry>& ograds) {
  using namespace nnvm;
  static const auto _backward_CachedOp = Op::Get("_backward_CachedOp");
  static const auto _NoGrad = Op::Get("_NoGradient");

  auto p = Node::Create();
  p->attrs.op = _backward_CachedOp;
  p->attrs.name = node->attrs.name + "_backward";
  p->attrs.parsed = node->attrs.parsed;
  p->control_deps.push_back(node);
  p->inputs.reserve(bwd_ograd_dep_.size() + bwd_in_dep_.size() + bwd_out_dep_.size());
  for (auto i : bwd_ograd_dep_) p->inputs.push_back(ograds[i]);
  for (auto i : bwd_in_dep_) p->inputs.push_back(node->inputs[i]);
  for (auto i : bwd_out_dep_) p->inputs.emplace_back(NodeEntry{node, i, 0});
  std::vector<NodeEntry> ret;
  ret.reserve(num_inputs());
  const auto& auxs = mutable_input_nodes();
  if (auxs.size()) {
    auto nop = Node::Create();
    nop->attrs.op = _NoGrad;
    nop->attrs.name = "NoGradient";
    uint32_t k = 0;
    for (const auto& i : fwd_graph_.indexed_graph().input_nodes()) {
      if (auxs.count(i)) {
        ret.emplace_back(NodeEntry{nop, 0, 0});
      } else {
        ret.emplace_back(NodeEntry{p, k++, 0});
      }
    }
  } else {
    for (uint32_t i = 0; i < num_inputs(); ++i) ret.emplace_back(NodeEntry{p, i, 0});
  }
  return ret;
}


bool CachedOp::SetForwardGraph(
    imperative::GraphRuntime* runtime,
    const bool recording,
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  CHECK_EQ(inputs.size(), num_inputs());
  nnvm::Graph& g = runtime->fwd_graph;

  ShapeVector shape_inputs;
  DTypeVector dtype_inputs;
  StorageTypeVector storage_type_inputs;
  shape_inputs.reserve(inputs.size());
  dtype_inputs.reserve(inputs.size());
  storage_type_inputs.reserve(inputs.size());
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    shape_inputs.emplace_back(inputs[i]->shape());
    dtype_inputs.emplace_back(inputs[i]->dtype());
    storage_type_inputs.emplace_back(inputs[i]->storage_type());
  }

  bool match = true;
  match &= CheckAndInferShape(&g, std::move(shape_inputs), true);
  match &= CheckAndInferType(&g, std::move(dtype_inputs), true);
  exec::DevMaskVector dev_mask(g.indexed_graph().num_nodes(), inputs[0]->ctx().dev_mask());
  match &= CheckAndInferStorageType(&g, std::move(dev_mask),
                                    std::move(storage_type_inputs), true);

  if (!match) {
    g.attrs.erase("forward_mem_plan");
    g.attrs.erase("full_mem_plan");
  } else if (g.attrs.count(recording ? "full_mem_plan" : "forward_mem_plan")) {
    return true;
  }

  const auto& idx = g.indexed_graph();

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
  CHECK_EQ(stypes.size(), storage.size());
  for (size_t i = 0; i < stypes.size(); i++) {
    if (stypes[i] != kDefaultStorage) storage[i] = exec::kDynamicStorageID;
  }
  for (const auto i : idx.input_nodes()) {
    storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  }
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    storage[idx.entry_id(idx.outputs()[i])] = exec::kExternalStorageID;
  }

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >(
          recording ? "full_ref_count" : "forward_ref_count"));
  g.attrs[recording ? "full_mem_plan" : "forward_mem_plan"] =
      std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;
}

bool CachedOp::SetBackwardGraph(
    imperative::GraphRuntime* runtime,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& inputs,
    bool detect_inplace_addto) {
  using namespace nnvm;
  using namespace imperative;
  std::lock_guard<std::mutex> lock(mutex_);
  Context default_ctx = inputs[0]->ctx();
  nnvm::Graph& g = runtime->full_graph;

  if (runtime->bwd_output_reqs != reqs) {
    runtime->bwd_output_reqs = reqs;
    runtime->bwd_input_eid.clear();
    g = nnvm::Graph();
    g.outputs = fwd_graph_.outputs;
    for (size_t i = 0; i < grad_graph_.outputs.size(); ++i) {
      if (runtime->bwd_output_reqs[i] == kNullOp) continue;
      g.outputs.emplace_back(grad_graph_.outputs[i]);
    }
    g.attrs["context"] = std::make_shared<dmlc::any>(
        std::vector<Context>(g.indexed_graph().num_nodes(), default_ctx));
  }

  const auto& idx = g.indexed_graph();

  if (runtime->bwd_input_eid.size() != inputs.size()) {
    runtime->bwd_input_eid.clear();
    for (const auto& i : bwd_ograd_dep_) {
      auto eid = idx.entry_id(ograd_entries_[i]);
      runtime->bwd_input_eid.push_back(eid);
    }
    for (const auto& i : bwd_in_dep_) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      runtime->bwd_input_eid.push_back(eid);
    }
    for (const auto& i : bwd_out_dep_) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      runtime->bwd_input_eid.push_back(eid);
    }
    CHECK_EQ(inputs.size(), runtime->bwd_input_eid.size());
  }

  if (!g.attrs.count("backward_ref_count")) {
    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes_; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }
    for (size_t i = 0; i < inputs.size(); ++i) ++ref_count[runtime->bwd_input_eid[i]];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    g.attrs["backward_ref_count"] = std::make_shared<dmlc::any>(std::move(ref_count));
  }

  auto shapes = runtime->fwd_graph.GetAttr<ShapeVector>("shape");
  shapes.resize(idx.num_node_entries(), TShape());
  auto dtypes = runtime->fwd_graph.GetAttr<DTypeVector>("dtype");
  dtypes.resize(idx.num_node_entries(), -1);
  auto stypes = runtime->fwd_graph.GetAttr<StorageTypeVector>("storage_type");
  stypes.resize(idx.num_node_entries(), -1);

  for (size_t i = 0; i < inputs.size(); ++i) {
    shapes[runtime->bwd_input_eid[i]] = inputs[i]->shape();
    dtypes[runtime->bwd_input_eid[i]] = inputs[i]->dtype();
    stypes[runtime->bwd_input_eid[i]] = inputs[i]->storage_type();
  }

  std::pair<uint32_t, uint32_t> node_range, entry_range;
  node_range = {num_forward_nodes_, idx.num_nodes()};
  entry_range = {num_forward_entries_, idx.num_node_entries()};

  bool match = true;
  match &= CheckAndInferShape(&g, std::move(shapes), false,
                              node_range, entry_range);
  match &= CheckAndInferType(&g, std::move(dtypes), false,
                             node_range, entry_range);
  exec::DevMaskVector dev_mask(idx.num_nodes(), default_ctx.dev_mask());
  match &= CheckAndInferStorageType(&g, std::move(dev_mask), std::move(stypes),
                                    false, node_range, entry_range);

  if (!match) {
    g.attrs.erase("backward_mem_plan");
  } else if (g.attrs.count("backward_mem_plan")) {
    return true;
  }

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  const auto& bwd_stypes = g.GetAttr<StorageTypeVector>("storage_type");
  for (size_t i = 0; i < bwd_stypes.size(); i++) {
    if (bwd_stypes[i] != kDefaultStorage) storage[i] = exec::kDynamicStorageID;
  }
  for (size_t i = 0; i < num_forward_entries_; ++i) storage[i] = exec::kExternalStorageID;
  for (const auto i : idx.input_nodes()) storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  for (const auto i : idx.outputs()) storage[idx.entry_id(i)] = exec::kExternalStorageID;

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >("backward_ref_count"),
      {num_forward_nodes_, idx.num_nodes()},
      {num_forward_entries_, idx.num_node_entries()},
      detect_inplace_addto);
  g.attrs["backward_mem_plan"] = std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;
}

OpStatePtr CachedOp::GetCachedOpState(
    const Context& ctx) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& i : cached_op_states_[ctx]) {
    // only create one state per device when not using static memory
    if (!config_.static_alloc || i.unique()) {
      return i;
    }
  }
  auto state_ptr = OpStatePtr::Create<CachedOpState>(ctx, fwd_graph_, full_graph_);

  cached_op_states_[ctx].push_back(state_ptr);
  return state_ptr;
}

std::multimap<size_t, NDArray> CachedOp::StaticAllocMemory(
    const Context& default_ctx,
    GraphRuntime *runtime,
    const GraphRuntime::Direction direction,
    std::multimap<size_t, NDArray>&& reuse_pool) {
  using namespace nnvm;
  using namespace imperative;

  nnvm::Graph& g = direction == GraphRuntime::kForward ?
      runtime->fwd_graph : runtime->full_graph;
  const auto& vstorage_inplace = g.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& mem_plan = g.GetAttr<MemoryPlanVector>(
      direction == GraphRuntime::kBackward ? "backward_mem_plan" :
          (runtime->is_recording ? "full_mem_plan" : "forward_mem_plan"));
  std::vector<int> addto_entry;
  if (g.attrs.count("addto_entry")) {
    addto_entry = g.GetAttr<std::vector<int> >("addto_entry");
  }
  auto range = runtime->GetRange(direction);
  const auto& idx = g.indexed_graph();

  for (auto i : idx.input_nodes()) {
    auto eid = idx.entry_id(i, 0);
    if (eid >= range.start_eid) runtime->dynamic_entries[eid] = true;
  }
  for (auto i : idx.outputs()) {
    auto eid = idx.entry_id(i);
    if (eid >= range.start_eid) runtime->dynamic_entries[eid] = true;
  }

  for (size_t i = range.start_eid; i < range.end_eid; ++i) {
    if (addto_entry.size() && addto_entry[i]) {
      runtime->array_reqs[i] = kAddTo;
    } else if (vstorage_inplace[i] >= 0) {
      runtime->array_reqs[i] = kWriteInplace;
    } else if (vstorage_inplace[i] == -2) {
      // -2 indicate that the entry is never referenced.
      runtime->array_reqs[i] = kNullOp;
    } else {
      runtime->array_reqs[i] = kWriteTo;
    }
  }

  reuse_pool = imperative::AllocateMemory(
      g, idx, default_ctx, range.start_eid, range.end_eid, mem_plan,
      runtime->arrays, &runtime->array_reqs, std::move(reuse_pool));
}

void CachedOp::StaticInitSeg(
    const Context& default_ctx,
    GraphRuntime *runtime,
    const GraphRuntime::Direction direction) {
  using namespace nnvm;
  using namespace imperative;

  nnvm::Graph& g = direction == GraphRuntime::kForward ?
      runtime->fwd_graph : runtime->full_graph;
  std::vector<int> skip_plus_node;
  if (g.attrs.count("skip_plus_node")) {
    skip_plus_node = g.GetAttr<std::vector<int> >("skip_plus_node");
  }
  auto range = runtime->GetRange(direction);
  const auto& idx = g.indexed_graph();

  if (!config_.static_shape) {
    for (size_t i = range.start_nid; i < range.end_nid; ++i) {
      runtime->segs[i].next_nid = i + 1;
      runtime->segs[i].skip = skip_plus_node.size() && skip_plus_node[i];
    }
    return;
  }

  for (size_t i = range.start_nid; i < range.end_nid; ++i) {
    exec::CreateOpExecs(g, &runtime->execs, i);
    runtime->execs[i]->op_ctx.is_train = runtime->is_training;
  }
  exec::AttachOpResources(g, runtime->execs, range.start_nid, range.end_nid);

  for (size_t i = range.start_nid; i < range.end_nid; ++i) {
    bool skip = idx[i].source->is_variable();
    for (size_t j = 0; !skip && j < idx[i].inputs.size(); ++j) {
      skip = runtime->dynamic_entries[idx.entry_id(idx[i].inputs[j])];
    }
    for (size_t j = 0; !skip && j < idx[i].source->num_outputs(); ++j) {
      skip = runtime->dynamic_entries[idx.entry_id(i, j)];
    }
    if (skip) continue;
    SetupOpExec(g, i, runtime->execs[i], runtime->arrays, runtime->array_reqs);
  }

  size_t bulk_size = idx.num_nodes();
  std::unordered_set<uint32_t> excludes;
  if (runtime->is_recording || direction == GraphRuntime::kBackward) {
    bulk_size = direction == GraphRuntime::kBackward ?
        config_.backward_bulk_size : config_.forward_bulk_size;
    for (const auto& i : idx.outputs()) excludes.insert(idx.entry_id(i));
    for (const auto& i : idx.input_nodes()) excludes.insert(idx.entry_id(i, 0));
  }

  CreateEngineOpSeg(idx, default_ctx, range.start_nid, range.end_nid, bulk_size,
                    excludes, runtime->execs, skip_plus_node, &runtime->segs);
}

void CachedOp::StaticRunOps(
    const Context& default_ctx,
    GraphRuntime *runtime,
    const GraphRuntime::Direction direction) {
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
  static auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");

  bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;

  nnvm::Graph& g = direction == GraphRuntime::kForward ?
      runtime->fwd_graph : runtime->full_graph;
  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
  GraphRange range = runtime->GetRange(direction);
  const auto& idx = g.indexed_graph();

  std::vector<NDArray*> ndinputs, ndoutputs;
  nnvm::ShapeVector arg_shapes;
  nnvm::DTypeVector arg_dtypes;
  std::vector<OpReqType> req;

  for (size_t i = range.start_nid; i < range.end_nid; i = runtime->segs[i].next_nid) {
    const auto& seg = runtime->segs[i];
    if (seg.skip) continue;
    if (seg.opr != nullptr) {
      Engine::Get()->Push(seg.opr.get(), default_ctx, 0, profiling);
    } else {
      const nnvm::IndexedGraph::Node& node = idx[i];
      if (node.source->is_variable()) continue;
      auto num_outputs = node.source->num_outputs();
      ndinputs.clear();
      ndinputs.reserve(node.inputs.size());
      for (const auto& j : node.inputs) {
        ndinputs.emplace_back(runtime->arrays[idx.entry_id(j)]);
        CHECK(!ndinputs.back()->is_none());
      }
      ndoutputs.clear();
      ndoutputs.reserve(num_outputs);
      req.clear();
      req.reserve(num_outputs);
      for (size_t j = 0; j < num_outputs; ++j) {
        size_t eid = idx.entry_id(i, j);
        ndoutputs.emplace_back(runtime->arrays[eid]);
        req.push_back(runtime->array_reqs[eid]);
        CHECK(req.back() == kNullOp || !ndoutputs.back()->is_none());
      }
      const DispatchMode dispatch_mode = dispatch_modes[i];
      if (createop.count(node.source->op())) {
        arg_shapes.clear();
        arg_dtypes.clear();
        arg_shapes.reserve(ndinputs.size());
        arg_dtypes.reserve(ndinputs.size());
        for (size_t i = 0; i < ndinputs.size(); ++i) {
          arg_shapes.emplace_back(ndinputs[i]->shape());
          arg_dtypes.emplace_back(ndinputs[i]->dtype());
        }
        runtime->op_states[i] = createop[node.source->op()](
            node.source->attrs, default_ctx, arg_shapes, arg_dtypes);
        Imperative::Get()->InvokeOp(
            default_ctx, node.source->attrs, ndinputs, ndoutputs, req,
            dispatch_mode, runtime->op_states[i]);
      } else if (is_layer_backward.get(node.source->op(), false)) {
        nnvm::Node* fwd_node = node.source->control_deps[0].get();
        auto fwd_node_id = idx.node_id(fwd_node);
        Imperative::Get()->InvokeOp(
            default_ctx, node.source->attrs, ndinputs, ndoutputs,
            req, dispatch_mode, runtime->op_states[fwd_node_id]);
      } else {
        Imperative::Get()->InvokeOp(
            default_ctx, node.source->attrs, ndinputs, ndoutputs, req,
            dispatch_mode);
      }
    }
  }
}

OpStatePtr CachedOp::StaticForward(
    const Context& default_ctx,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  bool recording = Imperative::Get()->is_recording();
  bool training = Imperative::Get()->is_training();
  auto state_ptr = GetCachedOpState(default_ctx);
  auto& state = state_ptr.get_state<CachedOpState>();
  GraphRuntime *runtime = &state.runtime;
  std::lock_guard<std::mutex> lock(state.mutex);

  bool match = SetForwardGraph(runtime, recording, inputs);
  match = match && runtime->is_recording == recording
      && runtime->is_training == training;
  nnvm::Graph& g = runtime->fwd_graph;
  const auto& idx = g.indexed_graph();

  if (!runtime->fwd_alloc || !match)  {
    runtime->ResetBuff(GraphRuntime::kForward, recording, training);
    state.fwd_reuse_pool = StaticAllocMemory(
        default_ctx, &state.runtime, GraphRuntime::kForward,
        std::move(state.fwd_reuse_pool));
  }

  if (config_.static_shape) {
    for (auto i : config_.param_indices) {
      auto nid = idx.input_nodes()[i];
      if (!runtime->arrays[idx.entry_id(nid, 0)]->IsSame(*inputs[i])) {
        match = false;
        auto ptr = &runtime->buff[idx.entry_id(nid, 0)];
        CHECK_EQ(runtime->arrays[idx.entry_id(nid, 0)], ptr);
        *runtime->arrays[idx.entry_id(nid, 0)] = *inputs[i];
        runtime->dynamic_entries[idx.entry_id(nid, 0)] = false;
      }
    }
    for (auto i : config_.data_indices) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      runtime->arrays[eid] = inputs[i];
    }
  } else {
    for (size_t i = 0; i < num_inputs(); ++i) {
      auto nid = idx.input_nodes()[i];
      runtime->arrays[idx.entry_id(nid, 0)] = inputs[i];
    }
  }

  if (!runtime->fwd_exec_init || !match) {
    runtime->ResetSeg(GraphRuntime::kForward);
    StaticInitSeg(default_ctx, runtime, GraphRuntime::kForward);
  }

  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& shapes = g.GetAttr<ShapeVector>("shape");
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    runtime->arrays[eid] = outputs[i];
    if (!outputs[i]->is_none()) continue;
    *outputs[i] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                          shapes[eid], default_ctx, true, dtypes[eid]);
  }

  StaticRunOps(default_ctx, runtime, GraphRuntime::kForward);

  return state_ptr;
}


OpStatePtr CachedOp::DynamicForward(
    const Context& default_ctx,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  // Initialize
  bool recording = Imperative::Get()->is_recording();
  bool training = Imperative::Get()->is_training();
  auto runtime_ptr = OpStatePtr::Create<GraphRuntime>();
  GraphRuntime *runtime = &runtime_ptr.get_state<GraphRuntime>();
  {
    auto state_ptr = GetCachedOpState(default_ctx);
    auto& state = state_ptr.get_state<CachedOpState>();
    std::lock_guard<std::mutex> lock(state.mutex);
    SetForwardGraph(&state.runtime, recording, inputs);
    runtime->fwd_graph = state.runtime.fwd_graph;
  }
  nnvm::Graph& g = runtime->fwd_graph;
  const auto& idx = g.indexed_graph();

  runtime->ResetBuff(GraphRuntime::kForward, recording, training);
  StaticAllocMemory(default_ctx, runtime, GraphRuntime::kForward);

  for (size_t i = 0; i < num_inputs(); ++i) {
    auto nid = idx.input_nodes()[i];
    runtime->arrays[idx.entry_id(nid, 0)] = inputs[i];
  }

  runtime->ResetSeg(GraphRuntime::kForward);
  StaticInitSeg(default_ctx, runtime, GraphRuntime::kForward);

  const auto& dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto& shapes = g.GetAttr<ShapeVector>("shape");
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");

  for (size_t i = 0; i < outputs.size(); ++i) {
  auto eid = idx.entry_id(idx.outputs()[i]);
  runtime->arrays[eid] = outputs[i];
  if (!outputs[i]->is_none()) continue;
  *outputs[i] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                        shapes[eid], default_ctx, true, dtypes[eid]);
  }

  StaticRunOps(default_ctx, runtime, GraphRuntime::kForward);

  return runtime_ptr;
}

void CachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  static const auto cached_op = nnvm::Op::Get("_CachedOp");

  CHECK_EQ(inputs.size(), num_inputs());

  Context default_ctx = inputs[0]->ctx();

  const auto& idx = fwd_graph_.indexed_graph();
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i]->ctx(), default_ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[0]].source->attrs.name
        << " is on " << default_ctx << " while "
        << idx[idx.input_nodes()[i]].source->attrs.name
        << " is on " << inputs[i]->ctx();
  }

  int prev_bulk_size = Engine::Get()->set_bulk_size(config_.forward_bulk_size);

  OpStatePtr op_state;
  if (config_.static_alloc) {
    op_state = StaticForward(default_ctx, inputs, outputs);
  } else {
    op_state = DynamicForward(default_ctx, inputs, outputs);
  }

  Engine::Get()->set_bulk_size(prev_bulk_size);

  if (Imperative::Get()->is_recording() && !inlining_) {
    nnvm::NodeAttrs attrs;
    attrs.op = cached_op;
    attrs.name = "_cachedop";
    attrs.parsed = op_ptr;
    Imperative::Get()->RecordOp(
        std::move(attrs), inputs, outputs, op_state,
        &save_inputs(), &save_outputs());
  }
}


void CachedOp::DynamicBackward(
    const bool retain_graph,
    const OpStatePtr& runtime_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  bool training = Imperative::Get()->is_training();
  Context default_ctx = outputs[0]->ctx();
  GraphRuntime *runtime = &runtime_ptr.get_state<GraphRuntime>();
  {
    auto state_ptr = GetCachedOpState(default_ctx);
    auto& state = state_ptr.get_state<CachedOpState>();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.runtime.fwd_graph = runtime->fwd_graph;
    SetBackwardGraph(&state.runtime, reqs, inputs);
    runtime->full_graph = state.runtime.full_graph;
    runtime->bwd_input_eid = state.runtime.bwd_input_eid;
  }
  nnvm::Graph& g = runtime->full_graph;
  const auto& idx = g.indexed_graph();

  runtime->ResetBuff(GraphRuntime::kBackward, true, training);
  StaticAllocMemory(default_ctx, runtime, GraphRuntime::kBackward);

  for (size_t i = 0; i < grad_graph_.outputs.size(); ++i) {
    auto entry = grad_graph_.outputs[i];
    if (!idx.exist(entry.node.get())) continue;
    auto eid = idx.entry_id(entry);
    runtime->array_reqs[eid] = reqs[i];
    runtime->arrays[eid] = outputs[i];
  }

  runtime->ResetSeg(GraphRuntime::kBackward);
  StaticInitSeg(default_ctx, runtime, GraphRuntime::kBackward);

  for (size_t i = 0; i < runtime->bwd_input_eid.size(); ++i) {
    auto eid = runtime->bwd_input_eid[i];
    if (runtime->dynamic_entries[eid]) runtime->arrays[eid] = inputs[i];
  }

  StaticRunOps(default_ctx, runtime, GraphRuntime::kBackward);
}

void CachedOp::StaticBackward(
    const bool retain_graph,
    const OpStatePtr& state_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  bool training = Imperative::Get()->is_training();
  auto& state = state_ptr.get_state<CachedOpState>();
  std::lock_guard<std::mutex> lock(state.mutex);

  Context default_ctx = state.context;
  GraphRuntime *runtime = &state.runtime;
  bool match = SetBackwardGraph(runtime, reqs, inputs, true);
  nnvm::Graph& g = runtime->full_graph;
  const auto& idx = g.indexed_graph();

  if (!runtime->bwd_alloc || !match) {
    runtime->ResetBuff(GraphRuntime::kBackward, true, training);
    state.bwd_reuse_pool = StaticAllocMemory(
        default_ctx, runtime, GraphRuntime::kBackward,
        std::move(state.bwd_reuse_pool));
  }

  if (config_.static_shape) {
    for (auto i : config_.param_indices) {
      const auto iter = fwd_input_to_grad_output_.find(i);
      if (iter == fwd_input_to_grad_output_.end()) continue;
      auto entry = grad_graph_.outputs[iter->second];
      if (!idx.exist(entry.node.get())) continue;
      auto eid = idx.entry_id(entry);
      if (!runtime->arrays[eid]->IsSame(*outputs[iter->second]) ||
          !(runtime->array_reqs[eid] == reqs[iter->second])) {
        match = false;
        runtime->array_reqs[eid] = reqs[iter->second];
        *runtime->arrays[eid] = *outputs[iter->second];
        runtime->dynamic_entries[eid] = false;
      }
    }
    for (auto i : config_.data_indices) {
      const auto iter = fwd_input_to_grad_output_.find(i);
      if (iter == fwd_input_to_grad_output_.end()) continue;
      auto entry = grad_graph_.outputs[iter->second];
      if (!idx.exist(entry.node.get())) continue;
      auto eid = idx.entry_id(entry);
      runtime->array_reqs[eid] = reqs[iter->second];
      runtime->arrays[eid] = outputs[iter->second];
    }
  } else {
    for (size_t i = 0; i < grad_graph_.outputs.size(); ++i) {
      auto entry = grad_graph_.outputs[i];
      if (!idx.exist(entry.node.get())) continue;
      auto eid = idx.entry_id(entry);
      runtime->array_reqs[eid] = reqs[i];
      runtime->arrays[eid] = outputs[i];
    }
  }

  if (!runtime->bwd_exec_init || !match) {
    runtime->ResetSeg(GraphRuntime::kBackward);
    StaticInitSeg(default_ctx, runtime, GraphRuntime::kBackward);
  }

  for (size_t i = 0; i < runtime->bwd_input_eid.size(); ++i) {
    auto eid = runtime->bwd_input_eid[i];
    if (runtime->dynamic_entries[eid]) runtime->arrays[eid] = inputs[i];
  }

  StaticRunOps(default_ctx, runtime, GraphRuntime::kBackward);
}

void CachedOp::Backward(
    const bool retain_graph,
    const OpStatePtr& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  using namespace imperative;
  CHECK(!Imperative::Get()->is_recording())
      << "CachedOp does not support higher order gradients. "
      << "If you want to do backward with create_graph=True please "
      << "do not use hybridize.";

  int prev_bulk_size = Engine::Get()->set_bulk_size(config_.backward_bulk_size);

  if (config_.static_alloc) {
    StaticBackward(retain_graph, state, inputs, reqs, outputs);
  } else {
    DynamicBackward(retain_graph, state, inputs, reqs, outputs);
  }

  Engine::Get()->set_bulk_size(prev_bulk_size);
}


NNVM_REGISTER_OP(_CachedOp)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(n->attrs.parsed);
    return op->Gradient(n, ograds);
  });

NNVM_REGISTER_OP(_backward_CachedOp)
.set_num_inputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_backward_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs){
    const CachedOpPtr& op = nnvm::get<CachedOpPtr>(attrs.parsed);
    return op->num_inputs() - op->mutable_input_nodes().size();
  })
.set_attr<bool>("TIsLayerOpBackward", true)
.set_attr<bool>("TIsBackward", true);

}  // namespace imperative
}  // namespace mxnet
