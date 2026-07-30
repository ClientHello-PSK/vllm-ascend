# HIXL vs Mooncake 迁移对比与 Block 寻址逻辑

> 对照 `MooncakeConnectorV1`（[`mooncake_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)，~3552 行）与 `HIXLConnectorV1`（[`hixl_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py)，~2787 行）。
> 生成日期：2026-07-28。代码行号为当前状态，实施前需复核。
> 配合 [`hixl-connector-design.md`](./hixl-connector-design.md)、[`hixl-connector-gap-design.md`](./hixl-connector-gap-design.md) 使用。

---

## 1. 功能点对比

### 1.1 已完成迁移（寻址无关 + 寻址适配）

| 类别 | mooncake 能力 | mooncake 锚点 | hixl 行号 | 适配方式 |
|---|---|---|---|---|
| **数据面注册** | `register_kv_caches`（register_memory 字节段） | :2332 | :2460 | `register_blocks_cache(CacheDesc, addrs, BlocksCacheKey, remote_accessible)` |
| 数据面传输 | `_transfer_kv_cache_all_groups`（batch_transfer_sync_read） | :774 | :714 | `pull_blocks(src/dst_blocks, layer_range)` |
| **控制面 ZMQ（P）** | `KVCacheSendingThread`（ROUTER） | :245 | :288 | 逐字 fork，换 `HixlAgentMetadata` |
| 控制面 ZMQ（D） | `KVCacheRecvingThread`（REQ 骨架） | :409 | :396 | 队列/socket 池/peer 公平 fork |
| 控制面 | `_get_remote_metadata` | :1367 | :670 | `ensure_linked` 替代缓存字节地址 |
| 控制面 | `_send_done_recv_signal` | :1401 | :961 | fork |
| 控制面 | `get_handshake_metadata`/`set_xfer_handshake_metadata[_pp_aware]` | :1592/:1605/:1617 | :1104/:1108/:1114 | fork |
| **几何 TP/PP** | `get_prefill_pp_indices` | :3752 | :108 | fork（module-level） |
| 几何 TP/PP | `_get_prefill_decode_size` | :2064 | :1619 | fork |
| 几何 TP/PP | `_get_remote_ranks_for_req`/`_get_remote_rank`/`_get_remote_tp_ranks` | :3574/:3544/:3547 | :1737/:1708/:1711 | fork，含 PP 维 |
| **几何 HMA** | `_get_hybrid_remote_rank_group_pulls` | :3197 | :1791 | fork，Mamba 真分支 |
| 几何 HMA | `_requires_group_aware_attention_transfer` | :2247 | :1641 | fork |
| 几何 HMA | `_get_attention_group_num_need_pulls[_for_decode_tp]` | :3255/:3258 | :1649/:1654 | fork |
| 几何 HMA | `_get_attention_group_num_key_value_heads`/`_get_attention_group_remote_rank` | :3269/:3285 | :1665/:1681 | fork |
| 几何 HMA | `_get_tp_num_need_pulls` | :3476 | :1697 | fork |
| **几何 CP** | `_get_group_pulls_metadata`/`make_group_pulls` | :3111/:3160 | :1933/:1962 | fork |
| 几何 CP | `_get_cp_shard_pulls` | :3060 | :1881 | fork（端口派生，寻址无关） |
| 几何 CP | `_get_kv_split_metadata`（No-CP） | :2662 | :1999 | block 级 |
| 几何 CP | `_get_kv_split_metadata` CP 分支 | :2720+ | :2037 | block 级（删 kernel 展开，G3） |
| 几何 CP | `_get_local_remote_cp_params` | :2635 | :1855 | fork，寻址无关 |
| **几何 block** | `_get_kernel_block_ids`（scale==1） | :2585 | :2360 | fork；scale>1=缺口 G1 |
| 几何 block | `_expand_block_ids`/`_group_compress_ratio`/`_get_kv_cache_group_id` | :2500/:2570/:2582 | :2339/:2345/:2357 | fork |
| **几何 group** | `_build_kv_group2layeridx` | :2179 | :1548 | fork，含 MTP/eagle/longcat |
| 几何 group | `_get_group_unique_specs`/`_get_group_transfer_info`/`_get_group_kv_caches` | :1699/:1680/:1196 | :1297/:1307/:848 | fork，含 mtp 归属 |
| **几何 MTP/Mamba** | `_get_transfer_block_ids` | :1710 | :1324 | fork，block 级裁剪 |
| 几何 Mamba | `_state_prefill_token_count`/`_truncate_request_for_prefill` | :1751/:1758 | :1195/:1202 | fork（末 token 重算） |
| 几何 Mamba | `_get_kernel_block_ids` Mamba align 分支 | :867 | :2368-2385 | final state block 选择 |
| 几何 compress | `_model_uses_compress` | :1675 | :1190 | fork（检测） |
| **多节点** | `_get_remote_host_info_by_port`/`multi_nodes_meta_mapping`/`set_xfer_handshake_metadata_from_workers` | :3491/:1941 | :2318/:1151/:1360 | fork |
| **生命周期** | `KVCacheTaskTracker`（delayed free + timeout） | :166 | :219 | 逐字 fork |
| 生命周期 | `shutdown`/`__init__` engine 句柄 | worker/:1977 | :2714/:1389 | `shutdown_datadist`/`get_datadist` 替代 `global_te` |
| **错误处理** | `_is/_mark/_clear_failed_recv_request`/`get_and_clear_invalid_block_ids`/`get_block_ids_with_load_errors` | :609/:613/:618/:602/:2494 | :538/:542/:547/:532/:2651 | fork |
| 错误处理 | `ret<0` 判错 | :896 | :819-824 | `LLMException` → `_mark_failed_recv_request` |
| **reformat（TP head）** | `reformat_kv_cache_hybrid_linear_torch` | :1083 | :915 | `_reformat_staging_to_local`（从 staging Cache 读，非 in-place） |
| reformat 机制 | `_stash_pending_reformat`/`_reformat_pending_kv_caches`/`_apply_kv_cache_reformat` | :994/:1003/:1014 | :866/:875/:885 | fork |
| **ZMQ helpers** | `zmq_ctx`/`ensure_zmq_send`/`recv`/`string_to_int64_hash`/`group_concurrent_contiguous` | :3637/:3709/:3730/:3699/:3652 | :2725/:2760/:2775/:2753/:2737 | fork |
| dispatcher | `HIXLConnector(KVConnectorBase_V1, SupportsHMA)` 全钩子 | :1506 | :1038 | fork，含 `request_finished_all_groups` |
| HMA 负载均衡 | `_set_hma_shared_port` | :2846 | :2150 | fork |

### 1.2 未完成迁移（缺口 + 原因）

| # | mooncake 能力 | mooncake 锚点 | hixl 状态 | 未完成原因 |
|---|---|---|---|---|
| 1 | **CP 多端口 done 信号** `_send_done_signal_to_free_remote_port` | :757 | ❌ 未迁移 | `start_load_kv`[:2694](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py#L2694) 调 `add_request` 未透传 `remote_port_send_num`（`_get_kv_split_metadata_cp`[:2199](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py#L2199) 算了但没传）。CP 多端口时 P 侧 done 计数错。**真实功能 bug，未文档化** |
| 2 | **MLA/compress `block_size_scale>1`** + `_get_group_kernel_params`/`_local_kernel_ids_for_shard`/MLA spec 序列化 | :2374/:2618/:2505/:2087 | ❌ `assert scale==1`（hixl:2528） | MLA/compress 把多 kernel block 压进一逻辑 block，需 kernel block 展开链路。归 MLA/compress 专项（[`gap-design`](./hixl-connector-gap-design.md) G1/G7/G8）。影响 DeepSeek-V3 |
| 3 | **NZ layout** `reformat_kv_cache` cat/nz 分支/`_cat_kv_cache`/`_nz_kv_cache`/fused op | :1244/:1321/:1347/:1218 | ❌ 仅 head 转置 | NPU NZ 排布需 `npu_scatter_pa_kv_cache`。归 NZ 专项（G3） |
| 4 | **SWA 滑窗裁剪** `_get_swa_transfer_block_ids`/`blocks_per_window`/`SlidingWindowSpec` | :1735/:142/:53 | ❌ 全缺 | 滑窗模型需窗口尾裁剪。归 SWA 专项（G4） |
| 5 | **sparse attention** `use_sparse` 检测 + HMA sparse 特例 | :2335/:2862 | ⚠️ 硬编码 `False`（hixl:1498） | sparse/MoE-attn（`index_topk`）需单 head group 退化。归 sparse 专项（G5） |
| 6 | **SFA DCP replicate-K** `_get_sfa_replicate_k_block_ids`/`local_full_block_ids`/`enable_sfa_dcp_replicated_indexer` | :3301/:127/:516/:916 | ❌ 全缺 | SFA + DCP + prefix cache 需 K 块跨 CP rank 复制。归 SFA 专项（G6/G10） |
| 7 | **conv_padding** `_get_mamba_conv_padding` + `base_addr -= conv_padding` | :2267/:2278-2293 | ❌ 全缺 | block 寻址无裸字节基址，不能搬 trick。MTP 草稿层若共享 mamba conv state 才需（Qwen3.5 草稿层纯 attention，非卡点，G2） |
| 8 | **`_prefill_get_remote_rank`/`_get_prefill_ranks_for_group`** | :3512/:3521 | ❌ 未迁移 | 疑为 P 侧反向拉取/SFA 场景辅助；HIXL P 全被动模型下似不需，**待核实** |
| 9 | **`_get_layer_spec`/`_get_kv_cache_dims_from_tensors`** | :2261/:1213 | ❌ 未迁移 | block 寻址下用 `ref_shape`/`kv_cache_config` 直接取，不需裸字节 dims（合理替代，非真缺口） |

> **迁移完整度**：寻址无关能力 ~98%，寻址方式适配 100%，能力缺口 G1-G10 0%（均归专项）。主线 Phase 1/2/3（除 MLA/compress）与 mooncake 高度对齐。

---

## 2. 已完成功能的 HIXL 适配（主要代码点）

### 2.1 数据面：注册 + 传输

**注册**（`register_kv_caches` [hixl:2460](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py#L2460)）：
- mooncake：`collect_storage_merged_register_regions` + `engine.register_memory(ptr, size)` 按字节段注册（HCCL region ≤256 合并）。
- hixl：收集 `kv_caches[layer_name]` 的 `data_ptr()` → `addrs`；`CacheDesc(num_tensors, shape=ref_shape, ...)` → `cache_manager.register_blocks_cache(cache_desc, addrs, BlocksCacheKey(cluster_id, model_id), remote_accessible=True)`。一个 group 一个 `Cache` 对象。
- **关键**：P/D 都 `remote_accessible=True`（hixl:2545）——llm_datadist 的 `PullCacheByGet` 路径（`EnableRemoteCacheAccessible=1`）要求 D 本地 dst cache 也 remote_accessible，否则 `pull_blocks` 返回 `LLM_PARAM_INVALID`（注释 hixl:2540-2544）。

**传输**（`_transfer_kv_cache_all_groups` [hixl:714](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py#L714)）：
- mooncake：`src = base_addr + block_id * stride + offset * inner_len`；`batch_transfer_sync_read(src_list, dst_list, length_list)`。
- hixl：`cache_manager.pull_blocks(BlocksCacheKey(remote_cluster_id, model_id), dst_cache, src_blocks, dst_blocks, src_layer_range, dst_layer_range)`。整 block 索引传输，无字节算术。
- `LLMException` → `_mark_failed_recv_request`（hixl:819-824）替代 mooncake `ret<0` 判错。

### 2.2 控制面：握手 + 建链

- mooncake：`session_id = f"{P_host}:{P_rpc_port}"` 隐式建链。
- hixl：`_get_remote_metadata`（hixl:670）解码 `HixlAgentMetadata` → **立即 `ensure_linked(remote_cluster_id, remote_ip, remote_port)`**（hixl:697-701）显式建链。
- 握手载荷：`MooncakeAgentMetadata`（字节字段 `te_rpc_port`/`kv_caches_base_addr`/`block_lens`/`block_strides`）→ `HixlAgentMetadata`（删字节字段，加 `cluster_id`/`listen_ip`/`listen_port`/`num_tensors_per_group`，hixl:134-155）。

### 2.3 TP>1 staging + reformat

- mooncake：字节偏移直写 D 真实 block 的 split i 位置 + 后置 `reformat` transpose。
- hixl：`_init_staging_caches`（hixl:2402）为 `tp_n>1` 的 attention group 分配 staging Cache（shape `[num_blocks*tp_n, block_size, head_per_split, dim]`，`remote_accessible=True`）；`_transfer` 的 `if tp_n>1 and not is_state_group`（hixl:767）走 staging：`dst_blocks = [b*tp_n + tp_offset]`，每 P rank shard 落 staging block `b*tp_n+i`；`_reformat_staging_to_local`（hixl:915）从 staging 读 → `reshape(N, tp_n, block, head_per_split, dim).transpose(1,2) → reshape(N, block, num_d_heads, dim)` → `index_copy_` 到 D 真实 cache。
- **Mamba state 不走 staging**：`is_state_group` 判定（hixl:765）让 state group 走 `else`（hixl:783）直落真实 cache（state 不 head-shard；且 vLLM hetero-TP 仅支持 `tp_ratio==1`，state 1对1 拉）。

### 2.4 MTP/Eagle（子项 A）

- `__init__` 识别 `speculative_config`（hixl:1406-1424）：MTP `method=="mtp"` → `num_draft_layers=1`；eagle → `draft_model_config.hf_config.num_hidden_layers`。
- `_build_kv_group2layeridx`（hixl:1548）mtp/eagle 层索引分支（hixl:1581-1588）：`"mtp" in name` → `idx = next_mtp_layer_idx`（从 `total_layers` 起）；eagle3 靠"层 id 已被占用"判定。
- `num_attn_module` 动态化（hixl:852/1569）：`2 if model_type=="longcat_flash" else 1`。
- `_get_transfer_block_ids`（hixl:1324）P 侧裁推测 block：attention group `blocks[:num_prompt_blocks]`，state group 不裁。

### 2.5 PP（子项 B，B3 偏离）

- 放开 `assert pp_size==1` + 加 `assert not(pp>1 and pcp>1)`（hixl:1434）。
- 建 `pp_layer_indices`（hixl:1451，fork mooncake :529）。
- **B3 偏离**：`_transfer` 的 `layer_range` 保持 `range(num_layers)`（hixl:807），**非** mooncake 的 `range(pp_first,pp_end)`。原因：vLLM PP 下每 rank 注册 cache 仅含本段层（`PPMissingLayer` + `_project_kv_cache_groups_to_worker`），pull 全部本段层即等价。`pp_layer_indices` 建而未用（保留供调试/未来）。

### 2.6 Mamba state（子项 C）

- `group_transfer_info`/`is_state_group`/`need_truncate`（hixl:1155/1307）：`is_state_group = any(MambaSpec)`。
- 末 token 重算：`_state_prefill_token_count`（hixl:1195，D 侧 N-1）+ `_truncate_request_for_prefill`（hixl:1202，P 丢末 token）+ `get_num_new_matched_tokens`（hixl:1160）。
- **Mamba align final state block**（`_get_kernel_block_ids` Mamba 分支 hixl:2368-2385）：align 模式 block table position-indexed 但只 `2+num_spec` 块常驻，故只拉 final resident block——remote 索引 `len - num_speculative_tokens - 1`，local 索引 0（fork mooncake :867）。
- `_append_mamba_transfer_meta` 显式 drop（hixl:1805-1811 注释）：block 寻址无字节算术对应物，state 由 `pull_blocks` 统一处理。

### 2.7 PCP/DCP（子项 D）

- worker 读 `pcp_size`/`pcp_rank`/`dcp_size`/`dcp_rank`（hixl:1426-1431）。
- `device_index = (pp_rank*pcp_size + pcp_rank)*tp_size + tp_rank`（hixl:1471/1526），`cluster_id`/`listen_port` 偏移含 pcp 维（hixl:1527）。
- `ReqMeta` 加 `remote_pcp_size`/`remote_dcp_size`（hixl:173-174），`request_finished` 回填（hixl:1288），`start_load_kv` 透传（hixl:2679）。
- `_get_cp_shard_pulls`（hixl:1881，fork :3060）：CP group_pulls 从端口派生（寻址无关）。
- `_get_kv_split_metadata_cp`（hixl:2037）：CP 分支 block 级——`get_local_remote_block_port_mappings`/`get_cp_group_meta`/`remote_block_nums_all` 轮询分配（fork :2774/:2748/:2932），attention 直接 block 切片（删 `_local_kernel_ids_for_shard` kernel 展开，G3）。
- **r_blk>1 assert**（hixl:2186）：`assert r_blk==1 or use_mla`。r_blk>1（Bd>Bp）需 MLA（scale>1 使 kernel_size 整除 Bp），mooncake scale=1 下亦不支持。

---

## 3. Block 寻址 vs 字节寻址 + HIXL Block 结构逻辑

### 3.1 寻址差异

| 维度 | mooncake（字节寻址） | hixl（block 寻址） |
|---|---|---|
| **注册单元** | 字节段 `register_memory(ptr, size)` | tensor `register_blocks_cache(CacheDesc, addrs, BlocksCacheKey)` |
| **元数据** | `kv_caches_base_addr[layer][cache]` + `block_len_per_addr` + `block_stride_per_addr`（裸 data_ptr/stride/len 三元组） | `CacheDesc(num_tensors, shape, data_type, placement)` + `BlocksCacheKey(cluster_id, model_id)` |
| **传输寻址** | `src = base_addr + block_id*stride + offset*inner_len`（字节偏移） | `pull_blocks(src_blocks, dst_blocks, src/dst_layer_range)`（block 索引） |
| **TP>1 head shard** | 字节偏移直写 D block 的 split i 位置 | staging block `b*tp_n + i` 整块落，后置 reformat transpose |
| **建链** | `session_id=f"{host}:{rpc_port}"` 隐式 | `ensure_linked(remote_cluster_id, ip, port)` 显式 |
| **路由 key** | `session_id` | `BlocksCacheKey(cluster_id, model_id)` |
| **错误** | `ret<0` | `LLMException` |
| **非连续处理** | `split_if_not_byte_contiguous` 字节级拆分 | `group_concurrent_contiguous` block 级合并 |

### 3.2 Block 结构逻辑：构造 → 生成 → 使用

#### 构造（register_kv_caches [hixl:2460-2530](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py#L2460)）

```
for group in kv_cache_groups:
    addrs = [t.data_ptr() for layer_name in group
             for t in _as_kv_cache_tuple(kv_caches[layer_name])]   # K/V 或 conv/ssm
    ref_shape = kv_caches[layer_name].shape                          # [num_blocks, block_size, *per_head_shape]
    cache_desc = CacheDesc(
        num_tensors = len(addrs),                  # = len(layer_indices) * 2  (K+V 或 conv+ssm)
        shape = ref_shape,                          # 完整单 tensor shape，含 num_blocks 维
        data_type = _torch_dtype_to_llm_dtype(...),
        placement = Placement.DEVICE,
    )
    cache = cache_manager.register_blocks_cache(
        cache_desc, addrs,
        BlocksCacheKey(cluster_id, model_id),       # 路由 key
        remote_accessible=True,                      # P/D 都 True（PullCacheByGet 要求）
    )
    group_caches[kv_cache_group_id] = cache
```

- `num_tensors`：每层 2 个 tensor（attention K+V 或 Mamba conv+ssm），`assert num_tensors == len(layer_indices)*2`（hixl:2491）。
- `shape[0] = num_blocks`（page 数），`shape[1] = block_size`（tokens/page），其后是 per-head shape。
- `scale = ref_shape[0] // self.num_blocks`，`assert scale==1`（hixl:2528，MLA/compress 才 >1）。

#### 生成（block_id 的语义）

- **block_id** = `CacheDesc.shape[0]` 维的索引（`0..num_blocks-1`），每个 block 是 `[block_size, *per_head_shape]` 个连续元素（一页）。
- **layer_range**：`layer i → tensor 索引 i*tensor_num_per_layer`（`layer_range_to_tensor_indices`，`hixl/llm_utils.py:318`）。默认 `range(0, num_tensors // tensor_num_per_layer)` = 全部层。
- **cluster_id**（pull key）：`cluster_id_base + dp_rank*(tp*pp*pcp) + (pp_rank*pcp_size + pcp_rank)*tp_size + tp_rank`（`_compute_identity` hixl:1516-1528）。P/D 用不相交 `cluster_id_base`（如 P=1000、D=2000）。
- **block_ids 来源**：`meta.local_block_ids`/`meta.remote_block_ids`（由 vLLM scheduler 的 block table 给出，position-indexed）。

#### 使用（_transfer [hixl:714-818](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py#L714)）

```
for group_pull in group_pulls:
    tp_n = group_pull.num_group_pulls
    src_blocks = remote_block_ids[kv_cache_group_id]     # P 的 block id 列表
    dst_logical = local_block_ids[kv_cache_group_id]     # D 的 block id 列表

    if tp_n > 1 and not is_state_group:                  # attention TP>1 → staging
        dst_cache = staging_caches[kv_cache_group_id]
        dst_blocks = [b*tp_n + tp_offset for b in dst_logical]   # shard i → staging block b*tp_n+i
    else:                                                # No-CP / Mamba state → 真实 cache
        dst_cache = group_caches[kv_cache_group_id]
        dst_blocks = dst_logical
        grouped_remote, grouped_local = group_concurrent_contiguous(src_blocks, dst_blocks)

    num_layers = dst_cache.cache_desc.num_tensors // 2
    src_layer_range = range(num_layers)                  # B3：本 rank 本段全部层
    dst_layer_range = range(num_layers)

    for chunk_remote, chunk_local in zip(grouped_remote, grouped_local):
        cache_manager.pull_blocks(
            BlocksCacheKey(remote_cluster_id, model_id),  # P 的路由 key
            dst_cache,
            src_blocks=chunk_remote, dst_blocks=chunk_local,
            src_layer_range=src_layer_range, dst_layer_range=dst_layer_range,
        )                                                # 整 block 传输，无字节偏移
    if tp_n > 1 and not is_state_group:
        stash_pending_reformat(...)                       # 收尾后 _reformat_staging_to_local
```

#### Mamba align 的 block 选择（特例）

- align 模式下 `meta.remote_block_ids[Mamba]` 是 position-indexed 长列表（覆盖 max_len），但只 `2+num_spec` 块有 live state，前部被 `remove_skipped_blocks` 置空。
- 故 `_get_kernel_block_ids` Mamba 分支（hixl:2368-2385）**只取 1 个 final state block**：`remote = block_ids[len - num_speculative_tokens - 1]`，`local = block_ids[0]`（fork mooncake :867）。不传整个 position-indexed 列表。

#### TP>1 head shard 拼接（staging）

- D 真实 block = `[block_size, num_d_heads, dim]`；P rank 的 block = `[block_size, head_per_split, dim]`（`head_per_split = num_d_heads/tp_n`）。
- `pull_blocks` 整 block 写，写不进 D block 的 split 子位置 → 用 staging：P rank i 的 shard 落 staging block `b*tp_n + i`（shape `[block_size, head_per_split, dim]`）。
- 所有 shard 落齐后，`_reformat_staging_to_local`（hixl:915）：`staging[N*tp_n, block, head_per_split, dim] → view[N, tp_n, block, head_per_split, dim] → transpose(1,2) → [N, block, tp_n, head_per_split, dim] → reshape[N, block, num_d_heads, dim] → index_copy_` 到 D 真实 block。

#### 路由（cluster_id 作为 pull key）

- P 侧 `register_blocks_cache` 用 `BlocksCacheKey(self.cluster_id, model_id)` 注册（暴露给远端）。
- D 侧 `pull_blocks` 用 `BlocksCacheKey(remote_cluster_id, model_id)` 拉取——`remote_cluster_id` 从 P 的 `HixlAgentMetadata.cluster_id` 取（`_get_remote_metadata` hixl:670）。
- `cluster_id` 全局唯一（P/D 不相交 base + rank offset），保证 D 能定位到对应 P rank 的 cache。

---

## 4. 总结

- **已完成**：数据面/控制面/几何（TP/PP/CP/HMA/MTP/Mamba/compress 检测）/生命周期/错误处理/TP-reformat/多节点/ZMQ/dispatcher 全部迁移且适配 block 寻址。
- **未完成**：CP 多端口 done 透传（真实 bug，待修）、MLA/compress（G1）、NZ（G3）、SWA（G4）、sparse（G5）、SFA（G6）、conv_padding（G2，Qwen3 非卡点）、`_prefill_get_remote_rank`（待核实）。
- **Block 寻址**：用 `CacheDesc`+`BlocksCacheKey`+`pull_blocks` 替代字节段+`base_addr+stride`，block_id 是 page 索引、layer_range 选层、cluster_id 路由；TP>1 用 staging block + reformat 拼头；Mamba align 只拉 final state block。

> 行号为 `vllm-ascend-v0.23.0` 当前状态，实施前需复核。MLA/compress 等专项详见 [`hixl-connector-gap-design.md`](./hixl-connector-gap-design.md)。
