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

  Graph& g = graph_;

  std::vector<NDArray*> full_inputs(num_forward_inputs_);
  for (index_t i = 0; i < fwd_input_idx_.size(); ++i) {
    full_inputs[fwd_input_idx_[i]] = inputs[i];
  }
  for (size_t i = 0; i < fwd_params_idx_.size(); ++i) {
    full_inputs[fwd_params_idx_[i]] = &params_[i];
  }
  bool match = SetupGraph(&g, config_.enable_backward, full_inputs);

  if (!initialized_ || !match) {
    Clear();
    g = exec::AttachOpExecs(g);
    g = exec::AttachOpResources(g);

    const auto& idx = g.indexed_graph();
    const auto& ref_count = g.GetAttr<std::vector<uint32_t> >("ref_count");
    const auto& mem_plan = g.GetAttr<MemoryPlanVector>("mem_plan");

    buff_.resize(idx.num_node_entries());
    arrays_.resize(idx.num_node_entries());
    for (size_t i = 0; i < idx.num_node_entries(); ++i) arrays_[i] = &buff_[i];
    for (size_t i = 0; i < fwd_params_idx_.size(); ++i) {
      auto nid = idx.input_nodes()[fwd_params_idx_[i]];
      arrays_[idx.entry_id(nid, 0)] = &params_[i];
    }

    array_reqs_.resize(idx.num_node_entries(), kWriteTo);
    for (size_t i = 0; i < idx.num_node_entries(); ++i) {
      if (ref_count[i] == 0) array_reqs_[i] = kNullOp;
    }

    imperative::AllocateMemory(
        g, idx, context_, 0, idx.num_node_entries(), mem_plan,
        arrays_, &array_reqs_);

    SetupCachedOps();

    initialized_ = true;
  }

  const auto& idx = g.indexed_graph();
  for (auto i : fwd_input_idx_) {
    auto eid = idx.entry_id(idx.input_nodes()[i], 0);
    arrays_[eid] = inputs[i];
  }
}

void Imperative::StaticCachedOp::StaticState::Forward(
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  static auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");

  std::lock_guard<std::mutex> lock(mutex_);

  bool recording = Imperative::Get()->is_recording();
  CHECK(config_.enable_backward || !recording)
      << "Set enable_backward to True to enable gradient calculation.";
  bool is_training = Imperative::Get()->is_training();

  Setup(inputs, outputs);

  nnvm::Graph& g = graph_;
  const auto& idx = g.indexed_graph();
  const auto& op_execs = g.GetAttr<exec::OpExecVector>("op_execs");
  const auto& dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");

  std::vector<NDArray*> ndinputs, ndoutputs;
  std::vector<OpReqType> req;

  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const nnvm::IndexedGraph::Node& node = idx[i];
    if (node.source->op() == nullptr) continue;
    if (oprs_[i] != nullptr) {
      bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
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

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    *outputs[i] = *arrays_[idx.entry_id(idx.outputs()[i])];
  }
}

Imperative::StaticCachedOp::StaticCachedOp(
    const CachedOpConfig& config,
    const nnvm::Symbol& sym,
    const std::vector<std::string> input_names,
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
    std::unordered_map<std::string, size_t> input_name_to_id;
    for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
      const auto& name = idx[idx.input_nodes()[i]].source->attrs.name;
      auto iter = parameters.find(name);
      if (iter == parameters.end()) {
        input_name_to_id[name] = i;
        continue;
      }
      fwd_params_idx_.push_back(i);
      for (const auto& param : iter->second) {
        params_[param.ctx()].emplace_back(param);
      }
    }

    CHECK_EQ(input_name_to_id.size(), input_names.size())
        << "Expecting " << input_name_to_id.size() << "inputs, given " << input_names.size();

    for (const auto& name : input_names) {
      auto iter = input_name_to_id.find(name);
      CHECK(iter != input_name_to_id.end()) << "Unexpected input name " << name;
      fwd_input_idx_.push_back(iter->second);
    }
  }

  if (!config_.enable_backward) return;

  // // construct backward graph
  // {
  //   const auto& idx = fwd_graph_.indexed_graph();
  //
  //   ograd_entries_.reserve(fwd_graph_.outputs.size());
  //   for (size_t i = 0; i < fwd_graph_.outputs.size(); ++i) {
  //     ograd_entries_.emplace_back(NodeEntry{Node::Create(), 0, 0});
  //   }
  //
  //   std::vector<NodeEntry> xs;
  //   std::vector<NodePtr> args = sym.ListInputs(Symbol::kReadOnlyArgs);
  //   xs.reserve(args.size());
  //   for (const auto& i : args) xs.emplace_back(NodeEntry{i, 0, 0});
  //   CHECK_GT(xs.size(), 0)
  //       << "There are no inputs in computation graph that require gradients.";
  //
  //   grad_graph_ = pass::Gradient(
  //       fwd_graph_, fwd_graph_.outputs, xs, ograd_entries_,
  //       exec::AggregateGradient, nullptr, nullptr,
  //       zero_ops, "_copy");
  // }
  //
  // // construct full graph
  // {
  //   size_t num_forward_nodes = fwd_graph_.indexed_graph().num_nodes();
  //   size_t num_forward_entries = fwd_graph_.indexed_graph().num_node_entries();
  //
  //   full_graph_.outputs = fwd_graph_.outputs;
  //   curr_grad_req_ = std::vector<bool>(grad_graph_.outputs.size(), true);
  //   for (const auto& i : grad_graph_.outputs) full_graph_.outputs.emplace_back(i);
  //   const auto& idx = full_graph_.indexed_graph();
  //
  //   std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
  //   for (size_t i = num_forward_nodes; i < idx.num_nodes(); ++i) {
  //     for (const auto& j : idx[i].inputs) {
  //        ++ref_count[idx.entry_id(j)];
  //     }
  //   }
  //
  //   auto full_ref_count = fwd_graph_.GetAttr<std::vector<uint32_t> >("forward_ref_count");
  //   for (size_t i = 0; i < num_forward_entries; ++i) full_ref_count[i] += ref_count[i];
  //   fwd_graph_.attrs["full_ref_count"] =
  //       std::make_shared<dmlc::any>(std::move(full_ref_count));
  //
  //   size_t num_forward_inputs = num_inputs();
  //   size_t num_forward_outputs = num_outputs();
  //   for (uint32_t i = 0; i < ograd_entries_.size(); ++i) {
  //     if (!idx.exist(ograd_entries_[i].node.get())) continue;
  //     auto eid = idx.entry_id(ograd_entries_[i]);
  //     if (ref_count[eid] > 0) {
  //       bwd_ograd_dep_.push_back(i);
  //     }
  //   }
  //   save_inputs_.resize(num_forward_inputs, false);
  //   for (uint32_t i = 0; i < num_forward_inputs; ++i) {
  //     auto eid = idx.entry_id(idx.input_nodes()[i], 0);
  //     if (ref_count[eid] > 0) {
  //       save_inputs_[i] = true;
  //       bwd_in_dep_.push_back(i);
  //     }
  //   }
  //   save_outputs_.resize(idx.outputs().size(), false);
  //   for (uint32_t i = 0; i < num_forward_outputs; ++i) {
  //     auto eid = idx.entry_id(idx.outputs()[i]);
  //     if (ref_count[eid] > 0) {
  //       save_outputs_[i] = true;
  //       bwd_out_dep_.push_back(i);
  //     }
  //   }
  // }
}

Context GetContext(
    const nnvm::IndexedGraph& idx,
    const std::vector<uint32_t> fwd_input_idx,
    const std::vector<NDArray*>& inputs) {
  Context ctx = inputs[0]->ctx();
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i]->ctx(), ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[fwd_input_idx[0]]].source->attrs.name << " is on "
        << ctx << " while " << idx[idx.input_nodes()[fwd_input_idx[i]]].source->attrs.name
        << " is on " << inputs[i]->ctx();
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
    state->graph_ = fwd_graph_;
    state->graph_.attrs["context"] = std::make_shared<dmlc::any>(
        std::vector<Context>(state->graph_.indexed_graph().num_nodes(), ctx));
    state->num_forward_inputs_ = idx.input_nodes().size();
    state->num_forward_outputs_ = idx.outputs().size();
    state->num_forward_nodes_ = idx.num_nodes();
    state->fwd_input_idx_ = fwd_input_idx_;
    state->fwd_params_idx_ = fwd_params_idx_;
    state->params_ = param_ptr->second;

    static_states_[ctx] = state;
    return state;
  }
  return state_iter->second;
}

void Imperative::StaticCachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  for (const auto& i : outputs)
    CHECK(i->is_none()) << "out must not be set when using static memory.";
  CHECK_EQ(inputs.size(), fwd_input_idx_.size())
      << "CachedOp requires " << fwd_input_idx_.size()
      << " inputs but got " << inputs.size();

  const auto& idx = fwd_graph_.indexed_graph();
  Context context = GetContext(idx, fwd_input_idx_, inputs);
  auto state = GetState(context);
  state->Forward(inputs, outputs);
}

}  // namespace mxnet
