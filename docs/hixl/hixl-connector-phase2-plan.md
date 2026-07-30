# HIXL Connector Phase 2 实施计划

> TP>1 staging + reformat；HMA 多 group。
> 配合 [`hixl-connector-design.md`](./hixl-connector-design.md)（背景+设计）与 [`hixl-connector-implementation.md`](./hixl-connector-implementation.md)（编码细节）使用。
> 生成日期：2026-07-24
> 代码引用用 `文件:行`（相对仓库根）。`hixl_connector.py` = [`vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py)；`mooncake_connector.py` = [`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)。

> **状态（2026-07-28）**：Phase 2 已完整落地（staging `remote_accessible=True` 非 §1.3/§4.4 所述 False；握手载荷未扩为列表，靠每 P rank 各发一份 HixlAgentMetadata）。本文为 Phase 2 计划历史记录，行号已偏移。最新状态见 [`hixl-connector-implementation.md` §0](./hixl-connector-implementation.md)。

---

## 1. 范围与现状

Phase 2 目标（设计文档 §3.8）：

- **TP>1 staging + reformat**：`pull_blocks` 只能整块写、写不进 split（设计文档 §2.2.3 / §3.3.3）。需在 D 端引入 staging，再后置 reformat 拼 head 维。
- **HMA 多 group**：混合架构（FullAttention + Mamba 等）按 group 分别注册/拉取。

### 1.1 Phase 1 留的口子（拦截点）

| 拦截 | 位置 | 含义 |
|---|---|---|
| `assert self.tp_size == 1` | `hixl_connector.py:1016` | TP>1 未支持 |
| `assert self.pp_size == 1` | `hixl_connector.py:1017` | Phase 3 才放开（本期不动） |
| `assert scale == 1`（block_size_scale） | `hixl_connector.py:1194-1197` | 只允许标准 FullAttention shape |
| `assert group_pull.num_group_pulls == 1` | `hixl_connector.py:645-647` | TP>1 staging 路径未实现 |
| reformat 是 no-op（仅注释） | `hixl_connector.py:688-690` | staging/reformat 全缺 |

### 1.2 已就绪、可复用的骨架

- `HIXLConnector(KVConnectorBase_V1, SupportsHMA)`（`hixl_connector.py:766`）已继承 `SupportsHMA`。
- `request_finished_all_groups`（`:797`）已转发；`HIXLConnectorScheduler.request_finished`（`:926`）内部已按多 group 循环（`:940`）—— **多 group 释放本就通**。
- `register_kv_caches`（`:1146`）已按 group 循环（`:1168`），一个 group 一个 `register_blocks_cache`（`:1212`）。
- ZMQ 控制面（`KVCacheSendingThread` / `KVCacheRecvingThread`）沿用，与引擎无关。
- `HixlAgentMetadata`（`:105`）已去字节寻址字段、含 `cluster_id`/`listen_*` 路由字段。

### 1.3 HIXL 相对 mooncake 缺失的关键件

| 件 | mooncake 锚点 | HIXL 状态 |
|---|---|---|
| block 几何 `_get_kv_split_metadata`（No-CP 分支） | `mooncake_connector.py:2662`（No-CP L2694-2718） | **整体删除** |
| `_get_group_pulls_metadata` / `make_group_pulls` | `:3111` / `:3160` | **无** |
| `_get_attention_group_num_need_pulls` | `:3255` | **无** |
| HMA 判定 `_is_hma_required`（属性，非方法）/ `_get_hybrid_remote_rank_group_pulls` | `:2020,2344` / `:3197` | **无** |
| staging 机制 `pending_reformat` / `_stash_pending_reformat` / `_reformat_pending_kv_caches` / `_apply_kv_cache_reformat` | `:469,994,1003,1014` | **无** |
| reformat 算子 `reformat_kv_cache` / `reformat_kv_cache_with_fused_op` / `reformat_kv_cache_hybrid_linear_torch` | `:1244,1218,1083` | **无** |
| `_handle_request` 的 `all_tasks_done` → reformat 分支 | `:729-745` | **删掉**（HIXL `:540` 附近） |
| `GroupPull` 字段 `remote_tp_offset` / `num_group_pulls` | `:144-150`（HIXL 已有定义） | **写死 0/1**（`:1323-1328`），下游不消费 |

> **关键认知（⚠ 与 mooncake 不可照搬）**：mooncake 的 reformat 走 in-place torch、不经过引擎 register，是因为它是**字节寻址**（`batch_transfer_sync_read` 的 dst 是裸 `data_ptr`）。HIXL 是 **block 索引寻址**，`pull_blocks(dst_cache, ...)` 的 `dst_cache` 必须是经 `register_blocks_cache` 注册的 Cache 对象（设计文档 §3.7）。因此 **staging 必须是注册的 Cache**（`remote_accessible=False`，本地 dst），pull 写入 staging Cache 后再用 torch reformat 拷到真实 `group_caches`——与设计文档 §3.3.3/§3.4 的"staging Cache"一致。**不要**把 mooncake 的 in-place torch 直接搬过来当 HIXL staging。

---

## 2. 交互与关键信息流

Phase 2 的改造本质是"让现有交互链路带上 TP>1 / 多 group 几何信息"。下面逐环节列出**传递了哪些关键字段**、**Phase 1 如何填**、**Phase 2 要补什么**。

### 2.1 启动期握手：worker → 本地 scheduler → 框架汇总

- **载荷**：`HixlAgentMetadata`（`hixl_connector.py:105-125`），worker 在 `register_kv_caches` 末尾组装（`:1228`）并存入 `self.xfer_handshake_metadata`（`:1246`）。
- **关键字段**：
  | 字段 | 含义 | Phase 1 | Phase 2 |
  |---|---|---|---|
  | `cluster_id` | 本 rank 的 LLMDataDist cluster_id，D 用作 pull 的 `BlocksCacheKey.cluster_id` | 单值 | **需扩为按 tp_rank 的列表**（D 要连 N 个 P rank） |
  | `listen_ip`/`listen_port` | D→P link 用 | 单值 | 同上，扩为列表 |
  | `model_id` | `BlocksCacheKey.model_id` | 0 | 不变 |
  | `num_tensors_per_group` | 每 group 的 `Cache.num_tensors`（=组内层数×2） | 已按 group | 不变 |
  | `kv_group2layeridx` | group→层映射 | 已多 group | 不变 |
  | `block_size`/`num_blocks` | block 几何 | 已传 | 不变 |
  | `block_size_scale` | logical/tensor block 比例 | 强制 `[1]`（`:1194`） | **放开 >1，真实传给 pull/reformat** |
  | `local_ip`/`handshake_port` | 多节点 host 映射 + ZMQ 端口 | 已传 | 不变 |
- **scheduler 侧整理**：`set_xfer_handshake_metadata_from_workers`（`:972`）把各 worker 握手整理成 `multi_nodes_meta_mapping{port_offset:{host,engine_id}}`（`:984`），供 P 端 `request_finished` 回填 `remote_multi_nodes_meta_mapping`。

### 2.2 P scheduler → D scheduler（经 disaggregated router，`kv_transfer_params`）

- **载荷**：P 端 `request_finished`（`:926`）返回的 `params_dict`（`:946-959`），经 vLLM 握手机制路由到 D，成为 D 请求的 `kv_transfer_params`（`do_remote_prefill=True`）。
- **关键字段**：
  | 字段 | 含义 | Phase 2 关联 |
  |---|---|---|
  | `remote_block_ids` | P 算出的 prompt 占的 block（已裁剪） | D pull 的 `src_blocks` 来源 |
  | `remote_engine_id`/`remote_request_id`/`remote_host`/`remote_port` | P 身份 + 握手端口基址 | D 端 `remote_handshake_port = remote_port + tp_rank`（`:1322`）需改为按多 rank |
  | `remote_ptp_size` | **P 的 TP size** | Phase 2 关键输入——D 据此算 `num_group_pulls = remote_ptp_size`（目前 `:954` 已传但下游未用） |
  | `remote_multi_nodes_meta_mapping` | 多节点 host 映射 | TP>1 多 rank 时按 tp_offset 索引取 host/port |
  | `num_prompt_blocks`/`remote_block_size` | block 几何 | block 展开用 |
  | `last_token_id` | 末 token（P→D 接力 decode） | 不变 |

### 2.3 D scheduler → D worker（经 `SchedulerOutput.kv_connector_metadata`）

- **载荷**：`HIXLConnectorMetadata`（`build_connector_meta` `:909`），含：
  - `requests: dict[req_id, ReqMeta]`（`ReqMeta` 见 `:128-141`）
  - `requests_to_send`（P 延迟发送的 req→时间戳）
  - `reqs_in_batch`（本批 req id 集）
- **`ReqMeta` 关键字段**：
  | 字段 | 含义 | Phase 2 |
  |---|---|---|
  | `local_block_ids`/`remote_block_ids` | D 分到的 block / P 的 block（均为 `BlockIds`，按 group 分组） | 多 group 天然多元素 |
  | `num_external_tokens`/`num_computed_tokens` | 加载/已算 token 数 | 不变 |
  | `remote_host`/`remote_port`/`remote_engine_id`/`remote_request_id` | P 身份 | 不变 |
  | `remote_ptp_size` | P 的 TP size | **Phase 2 须透传给 `start_load_kv` 算 `num_group_pulls`** |
  | `num_prompt_blocks`/`remote_block_size` | block 几何 | 不变 |

### 2.4 D worker `start_load_kv` → `KVCacheRecvingThread.add_request`

- **载荷**：`add_request` 的 `trans_info` dict（`:431-443`）。
- **关键字段**：
  | 字段 | Phase 1 | Phase 2 |
  |---|---|---|
  | `group_pulls: list[GroupPull]` | 写死 `GroupPull(num_group_pulls=1, remote_tp_offset=0, is_group_transfer_end=True)`（`:1323-1332`） | **用 `make_group_pulls` 生成真实 `num_group_pulls=remote_ptp_size`、每 rank 的 `remote_tp_offset`** |
  | `remote_handshake_port` | `meta.remote_port + self.tp_rank`（`:1322`，TP=1 即 +0） | **按 TP 几何算每个 P rank 的端口**（参考 MC `kv_port + dp*tp*pp*pcp + device_index`） |
  | `remote_port_send_num: dict[int, RemotePortInfo]` | **未传**（默认空 `{}`，`:425`）——这是预留的多 rank 多端口映射钩子 | **Phase 2 填 `{tp_rank: {num: port, host: ...}}`** |
  | `all_task_done` | `True`（单 rank 单 group 恒完成） | 多 rank 多 group 时仅**最后一个 group_pull** 置 `True`（驱动 `_handle_request` 的 `all_tasks_done` reformat 触发，对标 MC `:729-745`） |
  | `local_block_ids`/`remote_block_ids`/`num_computed_tokens` | 已传 | 不变 |
- **`GroupPull` 字段语义**（`:144-149`）：
  - `group_id`：组号
  - `remote_tp_offset`：本 group_pull 对应的 P rank 在 head split 里的偏移（reformat 拼 head 用）——Phase 1 恒 0，Phase 2 必须真实
  - `num_group_pulls`：该 group 要拉几次（= TP rank 数）——Phase 1 恒 1，Phase 2 = `remote_ptp_size`
  - `is_group_transfer_end`：是否该 group 最后一次拉（触发 reformat）——Phase 1 恒 True

### 2.5 RecvingThread → P SendingThread（ZMQ 控制面，两消息）

- **`GET_META_MSG`**（`:577`）：D 请求 P 的 `HixlAgentMetadata`。P 端 `KVCacheSendingThread` 回 `self.metadata`（fork 自 MC，仅换类型）。
  - Phase 2：D 收到后存 `remote_cluster_id[engine_id][handshake_port]`（`:602`）—— TP>1 需按 rank 存多个 cluster_id，每个 `ensure_linked`（`:595`）一次（HIXL link 单边语义，D 单边 link P，设计文档 §3.4）。
- **`DONE_RECVING_MSG`**（`:566`）：D 通知 P 拉完，载荷为 `remote_request_id`。P 端 `task_tracker.update_done_task_count`（`:192`）→ 取消 `delayed_free` 释放 block（`:197`）。

### 2.6 `pull_blocks` 调用（数据面，唯一真正传 KV）

- **签名**（`hixl_connector.py:670-677`）：
  ```
  cache_manager.pull_blocks(
      BlocksCacheKey(remote_cluster_id, model_id),  # 路由：连哪个 P rank
      dst_cache,                                      # D 端 Cache（group_caches[gid]）
      src_blocks=chunk_remote, dst_blocks=chunk_local,  # block 索引
      src_layer_range=range(num_layers), dst_layer_range=range(num_layers),
  )
  ```
- **Phase 1**：`dst_cache` 直接是 D 真实 cache；全 layer 一次拉；无 staging。
- **Phase 2**：`dst_cache` 改为 **staging 注册 Cache**（`register_blocks_cache`，`remote_accessible=False`，本地 dst）；`src_blocks`/`dst_blocks` 按 `remote_tp_offset` 落到 staging 的不同 block 段；拉完后 reformat 从 staging Cache tensor transpose 到真实 `group_caches` tensor。`remote_cluster_id` 按 rank 取（每个 P rank 一个 cluster_id）。
  > 注意：pull_blocks 的 dst 必须是 Cache 对象，不能用裸 tensor——这是 HIXL block 寻址与 mooncake 字节寻址的根本差异（§1.3）。
- **chunk 合并**：`group_concurrent_contiguous`（`:1374`）把连续 block 合并以减少 pull 调用数——Phase 2 多 group/多 rank 时仍适用，但 staging 落点计算须在合并前确定。

### 2.7 RecvingThread → D worker `post_forward`（回传 scheduler）

- **载荷**（经 `HIXLConnectorWorker` 转发）：
  - `get_finished`（`:1291`）→ `(finished_sending, finished_recving)`：来自 `task_tracker.get_and_clear_finished_requests`（`:447`）。scheduler 据此判定哪些 req 的 KV 转移真正结束、可释放延迟占用的 block。
  - `get_block_ids_with_load_errors`（`:1304`）→ `invalid_block_ids`：来自 `get_and_clear_invalid_block_ids`（`:449`）。D 拉取失败时 `_mark_failed_recv_request`（`:459`）标记，告诉调度器这些 block 的 KV 不可信、需重算。
- **Phase 2 注意（reformat 与 get_finished 的时序）**：reformat 在 `RecvingThread` 执行，`get_finished` 在主线程 `post_forward` 执行，两者靠 `task_tracker` 串行化——reformat 必须在 `_handle_request` 的 `try` 块里完成，`_mark_request_task_done` → `task_tracker.update_done_task_count`（`:560-561`，`finally` 块）**之后**才让 `get_finished` 可见该 req 完成。即现有 gating 天然保证"scheduler 看到 done ⇒ reformat 已拼完"，**不要**把 reformat 放到 `update_done_task_count` 之后。这也是 §4.5 把 reformat 挂在 `all_tasks_done` 分支的原因。

### 2.8 Phase 2 信息流增量总览

| 环节 | Phase 1 | Phase 2 增量 |
|---|---|---|
| 启动握手 `HixlAgentMetadata` | 单 `cluster_id`/`listen_*` | 扩为按 tp_rank 列表 |
| P→D `kv_transfer_params` | `remote_ptp_size` 已传未用 | D 据此算 `num_group_pulls` |
| `start_load_kv` group_pulls | 写死 1/0/True | `make_group_pulls` 生成真实 `num_group_pulls`/`remote_tp_offset` |
| `add_request` `remote_port_send_num` | 空 `{}` | 填多 rank 端口映射 |
| `add_request` `all_task_done` | 恒 True | 仅最后 group_pull 置 True |
| `_get_remote_metadata` | 单 rank `ensure_linked` | 多 rank 各 link 一次 |
| `pull_blocks` dst | D 真实 cache | staging 注册 Cache（`remote_accessible=False`） |
| reformat 触发 | 无 | `all_tasks_done` 分支调 `_reformat_pending_kv_caches` |

---

## 3. 实施清单（按依赖序）

| 步 | 内容 | HIXL 落点 | MC 对标 |
|---|---|---|---|
| 1 | 放开断言（tp_size / num_group_pulls；block_size_scale 条件性，见 §4.1） | `:1016 :645 (:1194)` | — |
| 2 | 移植 block 几何（仅 No-CP 分支） | worker 新增 | `:2694-2718, :3111` |
| 3 | `start_load_kv` 用真实几何替换写死 GroupPull | `:1309-1349`（写死在 `:1323-1328`） | `:3160 make_group_pulls` |
| 4 | staging（注册 Cache）+ reformat 机制 | RecvingThread 新增 | `:469,994,1003,1014` + `:1083,1218,1244` |
| 5 | `_handle_request` all_tasks_done 接入 reformat | `:540` 附近 | `:729-745` |
| 6 | HMA 算子（混合架构可选，见 §5 降级） | 新增 | `:2020,2344,3197` |
| 7 | 握手载荷扩展为多 P rank cluster_id 列表 | `HixlAgentMetadata` `:105` + `_get_remote_metadata` `:590` | — |
| 8 | 首个冒烟：staging→reformat 字节布局二次验证 | — | 设计文档 §3.8 末 |
| 9 | 与 MooncakeConnectorV1 TP>1 + HMA 逐位对齐 | — | 设计文档 §3.8 |

---

## 4. 各步展开

### 4.1 放开断言（步 1）

- `hixl_connector.py:1016`：`tp_size == 1` → 放开为 TP>1。
- `hixl_connector.py:645-647`：去掉 `num_group_pulls == 1`，改为按 `group_pull.num_group_pulls` 循环。
- `hixl_connector.py:1017` 的 `pp_size == 1` **本期不动**（Phase 3）。
- `hixl_connector.py:1194-1197` 的 `scale == 1`：**需先判定是否属 Phase 2 范围**——`block_size_scale` 衡量的是"tensor num_blocks / logical num_blocks"，即 MLA/compress 把多个 logical block 打包进一个 tensor block 的比例（MC `:2374`）。它与 **TP 无关**：TP>1 切的是 head 维（shape[2]），num_blocks（shape[0]）在各 rank 一致，标准 FullAttention 下 `scale` 仍为 1。
  - 若 Phase 2"HMA 多 group"只含 FullAttention + Mamba（不含 MLA/compress），则 **该 assert 本期不必放开**，staging/reformat 不依赖 `block_size_scale>1`。
  - 仅当引入 MLA/compress group 时才需放开，并把 `scale` 真正传给 pull/reformat（参考 MC `:916, :2600, :2626`）。**建议把"放开 block_size_scale"从 Phase 2 主线剥离**，归入 MLA/compress 专项（可能 Phase 3）。

### 4.2 移植 block 几何（步 2）

把 MC 的几何函数搬进 `HIXLConnectorWorker`，**只搬 No-CP 分支**（CP/PCP/DCP 留 Phase 3）：

- `_get_kv_split_metadata`（MC `:2662`，取 No-CP 分支 `:2694-2718`）：算"D 从哪些 P rank 拉哪些 block"。
- `_get_group_pulls_metadata`（MC `:3111`）+ `make_group_pulls`（MC `:3160`）：生成 `GroupPull` 列表（含真实 `remote_tp_offset` / `num_group_pulls`）。
- `_get_attention_group_num_need_pulls`（MC `:3255`）：按 group 算需要拉的 rank 数。

**HIXL 适配点（两条）**：

1. MC 这些函数输出字节地址算术（`src=base+block_id*stride`），HIXL 只需 block 索引——**删字节算术，保留 block 几何**。
2. ⚠ **`block_size_scale` 结构不匹配**：HIXL 现为 `block_size_scale[group]=[scale]`（每组单元素，`hixl_connector.py:1198`），而 MC 是 `block_size_scale[layer_idx][cache_idx]`（按层×cache 二维，MC `:2374-2379`）。MC 的 `_get_kernel_block_ids`（`:2585`）/`_get_group_kernel_params`（`:2618`）按 `[layer_idx][cache_idx]` 索引。**移植时须先统一结构**——要么把 HIXL 改成 layer×cache 二维，要么在移植函数里加 group→layer 索引转换。否则直接搬会越界。

### 4.3 `start_load_kv` 改造（步 3）

当前 `hixl_connector.py:1309-1349` 手写最小分片：

```
group_pulls = [GroupPull(group_id=gid, remote_tp_offset=0, num_group_pulls=1, ...) for ...]
remote_handshake_port = meta.remote_port + self.tp_rank   # 单 rank
```

改为调用步 2 移植的 `make_group_pulls` 生成真实 `group_pulls`，`remote_handshake_port` 按 TP 几何算每个 P rank 的端口（MC `:1956` 的 `handshake_port = kv_port + port_offset`；HIXL 自身 `side_channel_port` 基址在 `:869-874`）。

> ⚠ **TP>1 调用结构（易错点）**：`add_request` 的入参 `(remote_host, remote_handshake_port, group_pulls)` 针对的是**单个 peer（一个 P rank）**（`hixl_connector.py:424,431-443`）。TP>1 下 D 要连 N 个 P rank，因此 **`start_load_kv` 对一个请求要调 N 次 `add_request`**（每次一个 rank 的 host/port + 该 rank 的 group_pulls），**不是**一次调用塞 N 个 group_pull。仅**最后一次**调用传 `all_task_done=True`（驱动 `_handle_request` 在全部 rank 拉完后触发 reformat，对标 MC `:729-745`）。Phase 1 的单次调用 `:1334-1345`（`all_task_done=True`）只对 TP=1 成立。

### 4.4 staging + reformat（步 4，核心）

**staging（必须是注册 Cache）**：D 端经 `register_blocks_cache` 注册一个 staging Cache，`CacheDesc.shape=[logical_blocks * tp_n, block_size, head_per_split, dim]`、`remote_accessible=False`（本地 dst）。每个 P rank `pull_blocks` 到 staging Cache 的不同 block 段（`remote_tp_offset` 决定落点）。**不能**用裸 torch tensor 当 pull dst——HIXL block 寻址要求 dst 是 Cache 对象（见 §1.3 关键认知）。

**reformat**：从 staging Cache 的 tensor 做 `transpose(split, token)` 拷到 D 真实 `group_caches`。移植 MC 的三个算子（`reformat_kv_cache` `:1244` / `reformat_kv_cache_with_fused_op` `:1218` / `reformat_kv_cache_hybrid_linear_torch` `:1083`），纯 `torch_npu` 实现。这一步读 staging tensor、写真实 cache tensor（mooncake 此处是 in-place torch，HIXL 可沿用算子本身，但数据来源是 staging Cache 而非裸地址）。

**机制**：在 `KVCacheRecvingThread` 加回：

- `pending_reformat`（MC `:469`）
- `_stash_pending_reformat`（MC `:994`）
- `_reformat_pending_kv_caches`（MC `:1003`）
- `_apply_kv_cache_reformat`（MC `:1014`）

### 4.5 接入 reformat 到 `_handle_request`（步 5）

`hixl_connector.py:540` 附近的 `all_tasks_done` 分支，加回 MC `:729-745` 的 reformat 调用：所有 group pull 完成后触发 `_reformat_pending_kv_caches`。

### 4.6 握手载荷多 rank 扩展（步 7）

当前 `HixlAgentMetadata`（`:105`）只有单 `cluster_id`/`listen_ip`/`listen_port`。TP>1 下 D 要连 N 个 P rank：

- 把这三个字段改为按 tp_rank 的列表（或新增 `cluster_ids: list[int]` / `listen_infos: list[...]`）。
- `_get_remote_metadata`（`:590` 附近）按 rank 索引存储。
- `ensure_linked` 对每个 P rank 各调一次（HIXL link 单边语义，D 单边 link P，设计文档 §3.4）。

---

## 5. 降级路径

若 Phase 2 先只做 **多 FullAttention-style group + TP>1**（不做真正混合架构）：

- 步 6 的 HMA 算子可推迟到 Phase 3。
- 只走非-HMA 的 `_get_remote_rank`（MC `:3544`）+ `make_group_pulls`（MC `:3160`）路径。
- **步 2-5（staging + reformat）是硬核心，绕不开。**

---

## 6. 验收（设计文档 §3.8）

每期与 `MooncakeConnectorV1` 同 P/D 几何**逐位对齐**。Phase 2 三步：

1. **冒烟**：TP>1 staging pull + reformat 单 block 字节对齐（设计文档 §3.8 末明确要求的"二次验证"——`pull_blocks` 整块写 → staging → reformat 字节布局）。
2. **端到端**：`--kv-transfer-config kv_connector=HIXLConnectorV1`，TP>1 + HMA 多 group，与 mooncake 同配置 KV 逐位对齐（`External prefix cache hit rate: 100.0%`）。
3. **断言**：放开 Phase 1 四条 assert（`:645, :1016, :1017(PP 留 P3), :1194`）。

---

## 7. 风险点

| 风险 | 说明 | 缓解 |
|---|---|---|
| staging→reformat 字节布局 | 设计文档 §3.8 末点名的最大风险——pull_blocks 整块写无法直写 split，须经 staging Cache 中转 + reformat 拼 head | 最先做冒烟二次验证 |
| staging 须注册 Cache | 易误照搬 mooncake 的 in-place torch；HIXL block 寻址要求 pull dst 是 Cache 对象（§1.3） | staging 用 `register_blocks_cache(remote_accessible=False)`，reformat 再拷到真实 cache |
| `block_size_scale` 范围误判 | `scale>1` 是 MLA/compress 打包逻辑块所致，**与 TP 无关**；TP>1 FullAttention 下 scale 仍=1。误把放开该 assert 当 Phase 2 必做项会扩大范围 | 先确认 Phase 2 是否含 MLA/compress group；若仅 FullAttention+Mamba 则不必放开（§4.1） |
| `block_size_scale` 结构不匹配 | HIXL `block_size_scale[group]=[scale]` vs MC `block_size_scale[layer][cache]`；直接搬 MC 几何函数会越界 | 移植前先统一结构或加 group→layer 索引转换（§4.2） |
| TP>1 调用结构 | 易误用一次 `add_request` 塞 N 个 group_pull；实际一个 `add_request` = 一个 P rank，TP>1 须调 N 次 | `start_load_kv` 按 rank 循环调 `add_request`，仅最后一次 `all_task_done=True`（§4.3） |
| reformat 与 forward 时序 | reformat 写真实 cache 须在 forward 用该 KV 前完成 | reformat 在 `_handle_request` 的 `try` 块、`update_done_task_count` 之前；`get_finished` 经 task_tracker 天然 gate（§2.7） |
| 多 rank link 时序 | `remote_accessible=True` 的 register 必须在 link 之前（设计文档 §3.7） | P 被动不 link，D 端 link 顺序由 `_get_remote_metadata` 保证 |

> 注：本计划基于设计文档与当前代码静态分析，未在 NPU 上实测。行号为 `vllm-ascend-v0.23.0` 当前状态，实施前需复核（fork 源 mooncake 行号仅作对照锚点）。
