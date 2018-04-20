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
#include "../profiler/profiler.h"

namespace mxnet {
void Imperative::StaticCachedOp::StaticState::Clear() {
  initialized_ = false;
  buff_.clear();
  arrays_.clear();
  array_reqs_.clear();
  for (auto& op : oprs_) {
    if (op != nullptr) Engine::Get()->DeleteOperator(op);
  }
  oprs_.clear();
}

bool Imperative::StaticCachedOp::StaticState::SetupGraph(
    nnvm::Graph *graph,
    const bool enable_backward,
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  nnvm::Graph& g = *graph;

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
    g.attrs.erase("mem_plan");
  } else if (g.attrs.count("mem_plan")) {
    return true;
  }

  const auto& idx = g.indexed_graph();

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  for (const auto i : idx.input_nodes()) {
    storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  }
  for (size_t i = num_forward_outputs_; i < idx.outputs().size(); ++i) {
    storage[idx.entry_id(idx.outputs()[i])] = exec::kExternalStorageID;
  }
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
  CHECK_EQ(stypes.size(), storage.size());
  for (size_t i = 0; i < stypes.size(); i++) {
    if (stypes[i] != kDefaultStorage) storage[i] = exec::kDynamicStorageID;
  }

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >("ref_count"));
  g.attrs["mem_plan"] = std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;
}

void Imperative::StaticCachedOp::StaticState::SetupCachedOps() {
  // get the graph
  auto& g = graph_;
  const auto& idx = g.indexed_graph();
  const auto& vstorage_inplace = g.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& op_execs = g.GetAttr<exec::OpExecVector>("op_execs");

  std::vector<bool> skip_nodes(idx.num_nodes(), false);
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    for (const auto& e : inode.inputs) {
      if (arrays_[idx.entry_id(e)]->is_none()) {
        skip_nodes[nid] = true;
        break;
      }
    }
    for (uint32_t i = 0; i < inode.source->num_outputs(); ++i) {
      if (arrays_[idx.entry_id(nid, i)]->is_none()) {
        skip_nodes[nid] = true;
        break;
      }
    }
    if (skip_nodes[nid]) continue;

    auto& exec = op_execs[nid];
    CHECK_EQ(exec->in_array.size(), 0U);
    CHECK_EQ(exec->out_array.size(), 0U);
    for (const auto& e : inode.inputs) {
      exec->in_array.push_back(*arrays_[idx.entry_id(e)]);
    }
    // detect inplace requirement
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      exec->out_array.push_back(*arrays_[eid]);
      if (vstorage_inplace[eid] >= 0) {
        exec->req.push_back(kWriteInplace);
      } else if (vstorage_inplace[eid] == -2) {
        // -2 indicate that the entry is never referenced.
        exec->req.push_back(kNullOp);
      } else {
        exec->req.push_back(kWriteTo);
      }
    }
  }
  // Note that this modifies the requirment of kWriteInplace
  // for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) {
  //   auto& e = idx.outputs()[j];
  //   op_nodes_[e.node_id].exec->req[e.index] =
  //       grad_store_[j - num_forward_outputs_].first;
  // }
  oprs_.resize(idx.num_nodes(), nullptr);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (skip_nodes[nid]) continue;
    auto& exec = op_execs[nid];
    bool is_async = exec->exec_type() == ExecType::kAsync;
    bool is_gpu = context_.dev_mask() == gpu::kDevMask;

    // the variables
    std::vector<Engine::VarHandle> use_vars, mutate_vars;
    for (const auto& nd : exec->in_array) {
      use_vars.push_back(nd.var());
    }
    for (auto& r : exec->op_ctx.requested) {
      mutate_vars.push_back(r.var);
    }
    for (auto& nd : exec->out_array) {
      mutate_vars.push_back(nd.var());
    }
    if (exec->var() != nullptr) {
      mutate_vars.push_back(exec->var());
    }
    // dedup vars
    Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);
    // all vars include both mutate vars and use vars
    std::vector<Engine::VarHandle> all_vars(use_vars);
    std::copy(mutate_vars.begin(), mutate_vars.end(),
              std::inserter(all_vars, all_vars.end()));
    // setup exec vars
    Engine::Get()->PushAsync(
      [exec](RunContext rctx, Engine::CallbackOnComplete on_complete) {
        exec->Setup();
        on_complete();
      }, Context::CPU(), {}, all_vars, FnProperty::kNormal, 0, "SetupExec");
    auto exec_fun = [exec, is_async, is_gpu] (
        RunContext ctx, Engine::CallbackOnComplete on_complete) {
      if (is_async) {
        exec->op_ctx.async_on_complete = on_complete;
      }
      exec->Run(ctx, is_gpu);
      // call on complete only if it is async op
      if (!is_async) {
        if (is_gpu) {
        #if MXNET_USE_CUDA
          // Wait GPU kernel to finish.
          ctx.get_stream<gpu>()->Wait();
        #else
          LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
        #endif
        }
        on_complete();
      }
    };
    // setup the vars
    oprs_[nid] = Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, FnProperty::kNormal);
  }
}

void Imperative::StaticCachedOp::StaticState::Setup(
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace imperative;
  using namespace nnvm;

  bool match = SetupGraph(&graph_, config_.enable_backward, inputs);
  if (initialized_ && match) return;

  Graph& g = graph_;
  Clear();
  g = exec::AttachOpExecs(g);
  g = exec::AttachOpResources(g);

  const auto& idx = g.indexed_graph();
  const auto& ref_count = g.GetAttr<std::vector<uint32_t> >("ref_count");
  const auto& mem_plan = g.GetAttr<MemoryPlanVector>("mem_plan");

  buff_.resize(idx.num_node_entries());
  arrays_.resize(idx.num_node_entries());
  array_reqs_.resize(idx.num_node_entries(), kWriteTo);
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) array_reqs_[i] = kNullOp;
    arrays_[i] = &buff_[i];
  }
  for (size_t i = 0; i < fwd_params_idx_.size(); ++i) {
    auto nid = idx.input_nodes()[fwd_params_idx_[i]];
    arrays_[idx.entry_id(nid, 0)] = &params_[i];
  }
  if (config_.enable_backward) {
    for (size_t i = 0; i < fwd_params_idx_.size(); ++i) {
      const auto iter = fwd_in_to_bwd_out_.find(fwd_params_idx_[i]);
      if (iter == fwd_in_to_bwd_out_.end()) continue;
      LOG(INFO) << iter->second << " " << fwd_params_idx_[i] << " " << idx.outputs().size() << " " << num_forward_outputs_;
      auto eid = idx.entry_id(idx.outputs()[num_forward_outputs_ + iter->second]);
      LOG(INFO) << i << " " << eid << " " << arrays_.size() << " " << param_grads_.size();
      arrays_[eid] = &param_grads_[i];
    }
  }

  imperative::AllocateMemory(
      g, idx, context_, 0, idx.num_node_entries(), mem_plan,
      arrays_, &array_reqs_);

  SetupCachedOps();

  initialized_ = true;
}

void Imperative::StaticCachedOp::StaticState::RunOps(
    bool is_training, size_t start_nid, size_t end_nid) {
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");

  bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
  nnvm::Graph& g = graph_;
  const auto& idx = g.indexed_graph();
  const auto& op_execs = g.GetAttr<exec::OpExecVector>("op_execs");
  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");

  std::vector<NDArray*> ndinputs, ndoutputs;
  std::vector<OpReqType> req;

  for (size_t i = start_nid; i < end_nid; ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->op() == nullptr) continue;
    if (oprs_[i] != nullptr) {
      op_execs[i]->op_ctx.is_train = is_training;
      Engine::Get()->Push(oprs_[i], context_, 0, profiling);
    } else {
      auto num_outputs = node.source->num_outputs();
      ndinputs.clear();
      ndinputs.reserve(node.inputs.size());
      for (const auto& j : node.inputs) {
        ndinputs.emplace_back(arrays_[idx.entry_id(j)]);
        CHECK(!ndinputs.back()->is_none()) << idx[j.node_id].source->attrs.name << " " << j.index;
      }
      ndoutputs.clear();
      ndoutputs.reserve(num_outputs);
      req.clear();
      req.reserve(num_outputs);
      for (size_t j = 0; j < num_outputs; ++j) {
        size_t eid = idx.entry_id(i, j);
        ndoutputs.emplace_back(arrays_[eid]);
        req.push_back(array_reqs_[eid]);
        CHECK(!ndoutputs.back()->is_none());
      }
      const DispatchMode dispatch_mode = dispatch_modes[i];
      if (createop.count(node.source->op())) {
        Imperative::Get()->InvokeOp(
            context_, node.source->attrs, ndinputs, ndoutputs, req,
            dispatch_mode, op_execs[i]->state());
      } else {
        Imperative::Get()->InvokeOp(
            context_, node.source->attrs, ndinputs, ndoutputs, req,
            dispatch_mode);
      }
    }
  }
}

void Imperative::StaticCachedOp::StaticState::Forward(
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_training = Imperative::Get()->is_training();

  Setup(inputs, outputs);

  const auto& idx = graph_.indexed_graph();
  for (size_t i = 0; i < fwd_args_idx_.size(); ++i) {
    auto eid = idx.entry_id(idx.input_nodes()[fwd_args_idx_[i]], 0);
    arrays_[eid] = inputs[i];
  }

  RunOps(is_training, 0, num_forward_nodes_);

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    *outputs[i] = *arrays_[idx.entry_id(idx.outputs()[i])];
  }
}

void Imperative::StaticCachedOp::StaticState::Backward(
    const bool retain_graph,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_training = Imperative::Get()->is_training();

  const auto& idx = graph_.indexed_graph();
  for (size_t i = 0; i < bwd_input_eid_.size(); ++i) {
    arrays_[bwd_input_eid_[i]] = inputs[i];
  }
  for (size_t i = 0; i < fwd_args_idx_.size(); ++i) {
    const auto iter = fwd_in_to_bwd_out_.find(fwd_args_idx_[i]);
    if (iter == fwd_in_to_bwd_out_.end()) continue;
    auto eid = idx.entry_id(idx.outputs()[num_forward_outputs_ + iter->second]);
    arrays_[eid] = outputs[iter->second];
  }

  RunOps(is_training, num_forward_nodes_, idx.num_nodes());

}

Imperative::StaticCachedOp::StaticCachedOp(
    const CachedOpConfig& config,
    const nnvm::Symbol& sym,
    const std::vector<std::string> arg_names,
    const std::unordered_map<std::string, std::vector<NDArray> >& parameters)
      : config_(config) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const auto _copy = Op::Get("_copy");

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

    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    // TODO: no need to increment inputs?
    for (const auto& i : idx.input_nodes()) ++ref_count[idx.entry_id(i, 0)];
    for (const auto& i : idx.outputs()) ++ref_count[idx.entry_id(i)];
    for (size_t i = 0; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) ++ref_count[idx.entry_id(j)];
    }

    fwd_graph_.attrs["ref_count"] =
        std::make_shared<dmlc::any>(std::move(ref_count));
  }

  // Set parameters
  {
    const auto& idx = fwd_graph_.indexed_graph();
    std::unordered_map<std::string, size_t> arg_name_to_id;
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      const auto& name = idx[idx.input_nodes()[i]].source->attrs.name;
      auto iter = parameters.find(name);
      if (iter == parameters.end()) {
        arg_name_to_id[name] = i;
        continue;
      }
      fwd_params_idx_.push_back(i);
      for (const auto& param : iter->second) {
        params_[param.ctx()].emplace_back(param);
      }
    }

    CHECK_EQ(arg_name_to_id.size(), arg_names.size())
        << "Expecting " << arg_name_to_id.size() << "inputs, given " << arg_names.size();

    for (const auto& name : arg_names) {
      auto iter = arg_name_to_id.find(name);
      CHECK(iter != arg_name_to_id.end()) << "Unexpected input name " << name;
      fwd_args_idx_.push_back(iter->second);
    }
  }

  if (!config_.enable_backward) return;

  // construct backward graph
  {
    const auto& idx = fwd_graph_.indexed_graph();

    ograd_entries_.reserve(fwd_graph_.outputs.size());
    for (size_t i = 0; i < fwd_graph_.outputs.size(); ++i) {
      ograd_entries_.emplace_back(NodeEntry{Node::Create(), 0, 0});
    }

    std::vector<NodePtr> inputs = sym.ListInputs(Symbol::kAll);
    std::vector<NodeEntry> xs;
    xs.reserve(inputs.size());
    const auto& input_nodes = idx.input_nodes();
    const auto& mutable_nodes = idx.mutable_input_nodes();
    for (uint32_t i = 0; i < input_nodes.size(); ++i) {
      if (mutable_nodes.find(input_nodes[i]) != mutable_nodes.end()) continue;
      fwd_in_to_bwd_out_[i] = xs.size();
      xs.push_back(NodeEntry{inputs[i], 0, 0});
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
    size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
    size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();

    full_graph_.outputs = fwd_graph_.outputs;
    for (const auto& i : grad_graph_.outputs) full_graph_.outputs.emplace_back(i);
    const auto& idx = full_graph_.indexed_graph();

    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
      for (const auto& j : idx[i].inputs) {
         ++ref_count[idx.entry_id(j)];
      }
    }

    auto full_ref_count = fwd_graph_.GetAttr<std::vector<uint32_t> >("ref_count");
    // TODO: do this after save_inputs?
    for (size_t i = 0; i < num_forward_entries; ++i) full_ref_count[i] += ref_count[i];
    full_graph_.attrs["ref_count"] =
        std::make_shared<dmlc::any>(std::move(full_ref_count));

    size_t num_forward_inputs = num_inputs();
    size_t num_forward_outputs = num_outputs();
    for (uint32_t i = 0; i < num_forward_outputs; ++i) {
      if (!idx.exist(ograd_entries_[i].node.get())) continue;
      auto eid = idx.entry_id(ograd_entries_[i]);
      if (ref_count[eid] > 0) {
        bwd_ograd_dep_.push_back(i);
      }
    }
    save_inputs_.resize(num_forward_inputs, false);
    for (uint32_t i = 0; i < num_forward_inputs; ++i) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      if (ref_count[eid] > 0) {
        save_inputs_[i] = true;
        bwd_in_dep_.push_back(i);
      }
    }
    save_outputs_.resize(num_forward_outputs, false);
    for (uint32_t i = 0; i < num_forward_outputs; ++i) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      if (ref_count[eid] > 0) {
        save_outputs_[i] = true;
        bwd_out_dep_.push_back(i);
      }
    }
  }

  // calc bwd_input_eid_
  {
    const auto& idx = full_graph_.indexed_graph();
    for (const auto& i : bwd_ograd_dep_) {
      auto eid = idx.entry_id(ograd_entries_[i]);
      bwd_input_eid_.push_back(eid);
    }
    for (const auto& i : bwd_in_dep_) {
      auto eid = idx.entry_id(idx.input_nodes()[i], 0);
      bwd_input_eid_.push_back(eid);
    }
    for (const auto& i : bwd_out_dep_) {
      auto eid = idx.entry_id(idx.outputs()[i]);
      bwd_input_eid_.push_back(eid);
    }
  }
}

Context GetContext(
    const nnvm::IndexedGraph& idx,
    const std::vector<uint32_t> fwd_args_idx,
    const std::vector<NDArray*>& args) {
  Context ctx = args[0]->ctx();
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK_EQ(args[i]->ctx(), ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[fwd_args_idx[0]]].source->attrs.name << " is on "
        << ctx << " while " << idx[idx.input_nodes()[fwd_args_idx[i]]].source->attrs.name
        << " is on " << args[i]->ctx();
  }
  return ctx;
}

std::shared_ptr<Imperative::StaticCachedOp::StaticState>
Imperative::StaticCachedOp::GetState(
    const Context& ctx) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto state_iter = static_states_.find(ctx);
  if (state_iter == static_states_.end()) {
    auto param_ptr = params_.find(ctx);
    CHECK(param_ptr != params_.end()) << "CachedOp was not initialized on " << ctx;

    auto state = std::make_shared<StaticState>();
    state->config_ = config_;
    state->context_ = ctx;

    const auto& idx = fwd_graph_.indexed_graph();
    state->num_forward_inputs_ = idx.input_nodes().size();
    state->num_forward_outputs_ = idx.outputs().size();
    state->num_forward_nodes_ = idx.num_nodes();

    state->graph_ = config_.enable_backward ? full_graph_ : fwd_graph_;
    state->graph_.attrs["context"] = std::make_shared<dmlc::any>(
        std::vector<Context>(state->graph_.indexed_graph().num_nodes(), ctx));
    state->fwd_args_idx_ = fwd_args_idx_;
    state->fwd_params_idx_ = fwd_params_idx_;
    state->fwd_in_to_bwd_out_ = fwd_in_to_bwd_out_;
    state->bwd_input_eid_ = bwd_input_eid_;
    state->params_ = param_ptr->second;
    for (const auto& i : state->params_) state->param_grads_.emplace_back(i.grad());

    static_states_[ctx] = state;
    return state;
  }
  return state_iter->second;
}

void Imperative::StaticCachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& args,
    const std::vector<NDArray*>& outputs) {
  for (const auto& i : outputs)
    CHECK(i->is_none()) << "out must not be set when using static memory.";
  CHECK_EQ(args.size(), fwd_args_idx_.size())
      << "CachedOp requires " << fwd_args_idx_.size()
      << " inputs but got " << args.size();
  bool recording = Imperative::Get()->is_recording();
  CHECK(config_.enable_backward || !recording)
      << "Set enable_backward to True to enable gradient calculation.";

  Context context = GetContext(fwd_graph_.indexed_graph(), fwd_args_idx_, args);
  auto state = GetState(context);

  std::vector<NDArray*> inputs(num_inputs());
  for (index_t i = 0; i < fwd_args_idx_.size(); ++i) {
    inputs[fwd_args_idx_[i]] = args[i];
  }
  for (size_t i = 0; i < fwd_params_idx_.size(); ++i) {
    inputs[fwd_params_idx_[i]] = &state->params_[i];
  }
  state->Forward(inputs, outputs);

  if (recording) {
    nnvm::NodeAttrs attrs;
    attrs.op = nnvm::Op::Get("_CachedOp");
    attrs.name = "_cachedop";
    attrs.parsed = op_ptr;
    Imperative::Get()->RecordOp(
        std::move(attrs), inputs, outputs, OpStatePtr(),
        &save_inputs_, &save_outputs_);
  }
}

void Imperative::StaticCachedOp::Backward(
    const bool retain_graph,
    const OpStatePtr& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<OpReqType>& reqs,
    const std::vector<NDArray*>& outputs) {

  Context context = inputs[0]->ctx();  // TODO: fix
  auto static_state = GetState(context);

  static_state->Backward(retain_graph, inputs, reqs, outputs);
}

std::vector<nnvm::NodeEntry> Imperative::StaticCachedOp::Gradient(
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

}  // namespace mxnet
