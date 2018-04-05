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

namespace mxnet {

Imperative::StaticCachedOp::StaticCachedOp(
    const nnvm::Symbol& sym,
    const std::vector<std::pair<std::string, std::string> >& kwargs,
    const std::vector<Context> contexts,
    const std::vector<std::string> input_names,
    const std::unordered_map<std::string, std::vector<NDArray> >& parameters) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const auto _copy = Op::Get("_copy");

  param_.Init(kwargs);

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

  CHECK_GE(contexts.size(), 1)
      << "Contexts must be set when static_memory is true";

  for (const auto& ctx : contexts) {
    static_states_[ctx] = std::make_shared<StaticState>();
    static_states_[ctx]->context = ctx;
  }

  const auto& idx = fwd_graph_.indexed_graph();
  std::unordered_map<std::string, size_t> input_name_to_id;
  for (size_t i = 0; i < idx.input_nodes().size(); ++i) {
    const auto& name = idx[idx.input_nodes()[i]].source->attrs.name;
    auto iter = parameters.find(name);
    if (iter == parameters.end()) {
      input_name_to_id[name] = i;
      continue;
    }
    for (const auto& param : iter->second) {
      static_states_[param.ctx()]->params.emplace_back(i, param);
    }
  }

  CHECK_EQ(input_name_to_id.size(), input_names.size())
      << "Expecting " << input_name_to_id.size() << "inputs, given " << input_names.size();

  for (const auto& name : input_names) {
    auto iter = input_name_to_id.find(name);
    CHECK(iter != input_name_to_id.end()) << "Unexpected input name " << name;
    input_idx_.push_back(iter->second);
  }

  for (const auto& ctx : contexts) {
    static_states_[ctx]->graph = fwd_graph_;
    static_states_[ctx]->graph.attrs["context"] = std::make_shared<dmlc::any>(std::vector<Context>(idx.num_nodes(), ctx));
  }
}

bool Imperative::StaticCachedOp::SetupGraph(
    nnvm::Graph *graph,
    const bool enable_backward,
    const std::vector<NDArray*>& inputs) {
  using namespace nnvm;
  using namespace imperative;
  CHECK_EQ(inputs.size(), num_inputs());
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

void Imperative::StaticCachedOp::SetupCachedOps(
    const std::shared_ptr<StaticState>& state) {
  // get the graph
  auto& g = state->graph;
  const auto& idx = g.indexed_graph();
  const auto& vstorage_inplace = g.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& op_execs = g.GetAttr<exec::OpExecVector>("op_execs");

  std::vector<bool> skip_nodes(idx.num_nodes(), false);
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    for (const auto& e : inode.inputs) {
      if (state->arrays[idx.entry_id(e)]->is_none()) {
        skip_nodes[nid] = true;
        break;
      }
    }
    if (skip_nodes[nid]) continue;

    auto& exec = op_execs[nid];
    CHECK_EQ(exec->in_array.size(), 0U);
    CHECK_EQ(exec->out_array.size(), 0U);
    for (const auto& e : inode.inputs) {
      exec->in_array.push_back(*state->arrays[idx.entry_id(e)]);
    }
    // detect inplace requirement
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      exec->out_array.push_back(*state->arrays[eid]);
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
  state->oprs.resize(idx.num_nodes(), nullptr);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (skip_nodes[nid]) continue;
    auto& exec = op_execs[nid];
    bool is_async = exec->exec_type() == ExecType::kAsync;
    bool is_gpu = state->context.dev_mask() == gpu::kDevMask;

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
    state->oprs[nid] = Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, FnProperty::kNormal);
  }
}

void Imperative::StaticCachedOp::SetupState(
    const std::shared_ptr<StaticState>& state,
    const std::vector<NDArray*>& inputs,
    const std::vector<NDArray*>& outputs) {
  using namespace imperative;
  using namespace nnvm;

  Graph& g = state->graph;

  bool match = SetupGraph(&g, param_.enable_backward, inputs);
  if (match && state->initialized) return;

  g = exec::AttachOpExecs(g);
  g = exec::AttachOpResources(g);

  const auto& idx = g.indexed_graph();
  const auto& ref_count = g.GetAttr<std::vector<uint32_t> >("ref_count");
  const auto& mem_plan = g.GetAttr<MemoryPlanVector>("mem_plan");

  state->buff.clear();
  state->buff.resize(idx.num_node_entries());
  state->arrays.resize(idx.num_node_entries());
  for (size_t i = 0; i < idx.num_node_entries(); ++i) state->arrays[i] = &state->buff[i];
  for (auto& kv : state->params) {
    state->arrays[idx.entry_id(idx.input_nodes()[kv.first], 0)] = &kv.second;
  }

  state->array_reqs.clear();
  state->array_reqs.resize(idx.num_node_entries(), kWriteTo);
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0) state->array_reqs[i] = kNullOp;
  }

  imperative::AllocateMemory(
      g, idx, state->context, 0, idx.num_node_entries(), mem_plan,
      state->arrays, &state->array_reqs);

  SetupCachedOps(state);

  state->initialized = true;
}

Context GetDefaultContext(
    const nnvm::IndexedGraph& idx,
    const std::vector<NDArray*>& inputs) {
  Context default_ctx = inputs[0]->ctx();
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i]->ctx(), default_ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[0]].source->attrs.name << " is on " << default_ctx
        << " while " << idx[idx.input_nodes()[i]].source->attrs.name << " is on "
        << inputs[i]->ctx();
  }
  return default_ctx;
}

void Imperative::StaticCachedOp::Forward(
    const std::shared_ptr<CachedOp>& op_ptr,
    const std::vector<NDArray*>& partial_inputs,
    const std::vector<NDArray*>& outputs) {

  CHECK_EQ(partial_inputs.size(), input_idx_.size())
      << "CachedOp requires " << input_idx_.size()
      << " inputs but got " << partial_inputs.size();
  auto& state = static_states_[partial_inputs[0]->ctx()];
  std::lock_guard<std::mutex> lock(state->mutex);

  std::vector<NDArray*> inputs(fwd_graph_.indexed_graph().input_nodes().size());
  for (auto& kv : state->params) {
    inputs[kv.first] = &kv.second;
  }
  for (index_t i = 0; i < input_idx_.size(); ++i) {
    inputs[input_idx_[i]] = partial_inputs[i];
  }

  for (const auto& i : outputs)
    CHECK(i->is_none()) << "out must not be set when using static memory.";

  bool recording = Imperative::Get()->is_recording();
  CHECK(param_.enable_backward || !recording)
      << "Set enable_backward to True to enable gradient calculation.";
  SetupState(state, inputs, outputs);
  nnvm::Graph& g = state->graph;
  const auto& idx = g.indexed_graph();

  Context default_ctx = GetDefaultContext(idx, inputs);


}

}  // namespace mxnet
