# HIXL Connector 开发实现细节

> 编码参考，配合 [`hixl-connector-design.md`](./hixl-connector-design.md)（背景+设计）使用。design 文档讲"为什么这么设计"，本文讲"具体怎么写代码"。
> 生成日期：2026-07-17
> 代码引用用 `文件:行`（相对各自仓库根）。fork 源 `mooncake_connector.py`（3552 行）；Phase 1 实现 `hixl_connector.py`（~780 行）。

---

## 0. 状态更新（2026-07-28，Phase 3 完成后）

> 本文原写于 Phase 1（2026-07-17，~780 行）。Phase 2/3 已全部落地，当前 `hixl_connector.py` ~2785 行。下方 §6/§7 为 Phase 1 历史状态，**已过时**；以本节为准。

- **已实施**：Phase 2（TP>1 staging+reformat、HMA 多 group）、Phase 3（A MTP/Eagle、B PP、C Mamba state、D PCP/DCP）。
- **当前 assert 约束**（替代 §6.2）：仅保留 `scale==1`（行 2527，MLA/compress 专项）、`r_blk==1 or use_mla`（行 2185，r_blk>1 限 MLA）、`not(pp>1 and pcp>1)`（行 1433）、`prefill_tp>=decode_tp`（行 1635）。`tp_size==1`/`pp_size==1`/`num_group_pulls==1` assert 均已删除。
- **三处设计偏离/修复**（文档原未记录）：
  1. **B3 layer_range**：保留 `range(num_layers)`（行 806），非计划 §3.5 的 `range(pp_first,pp_end)`。因每 PP rank 注册 cache 仅含本段层（vLLM PP 隔离），pull 全部本段层即等价。`pp_layer_indices` 建而未用。
  2. **r_blk>1 assert**（行 2185）：r_blk>1（Bd>Bp）限 MLA/compress（scale>1 使 kernel_size 整除 Bp），非 MLA 下 fail fast。mooncake 同样不支持 scale=1 下 r_blk>1。
  3. **Mamba align final state block**（行 2367-2384，`_get_kernel_block_ids` Mamba 分支）：align 模式只拉 final resident state block（remote 索引 `len-num_speculative_tokens-1`，local 索引 0，fork mooncake :867-869）。
- **staging remote_accessible**：实际为 `True`（行 2451），非 §5/§3.2 所述 False。llm_datadist PullCacheByGet 路径要求 dst cache 也 remote_accessible（行 2462-2467 注释）。
- **`_append_mamba_transfer_meta`**：显式 drop（行 1809-1810 注释），block 寻址无对应物，Mamba state 由 `_transfer` 的 `pull_blocks` 统一处理。
- **剩余缺口**：见 [`hixl-connector-gap-design.md`](./hixl-connector-gap-design.md)（MLA/compress、SWA、sparse、NZ、SFA、conv_padding 等）。
- **行号**：本文及他文档的 `:行号` 引用均为 Phase 1/2 旧值，Phase 3 后已偏移，以代码为准。

---

## 1. 文件与符号清单（fork 策略）

fork 自 `mooncake_connector.py`。完整 V1 目标的符号处理：

| 符号 | mooncake 锚点 | 处理 |
|---|---|---|
| imports | :1-79 | 改：去 `from mooncake.engine import TransferEngine`；加 `from vllm_ascend...utils.hixl_datadist import get_datadist, shutdown_datadist`；`llm_datadist` 的 `CacheDesc/BlocksCacheKey/Placement/DataType` 在方法内懒加载 import |
| 常量 `GET_META_MSG/DONE_RECVING_MSG` 等 | :81-88 | 逐字拷贝 |
| `RemotePortInfo` / `ReqMeta` / `GroupPull` / `GroupTransferInfo` / `SizedDict` / `KVCacheTaskTracker` | :91-241 | **逐字拷贝**（引擎无关） |
| **`HixlAgentMetadata(msgspec.Struct)`** | :96-107 | **重写**（见 §2） |
| `HIXLConnectorMetadata` | :1380-1408 | 逐字拷贝（改类名 + ReqMeta 引用） |
| ZMQ helpers（`zmq_ctx`/`ensure_zmq_send/recv`/`string_to_int64_hash`/`group_concurrent_contiguous`） | :3420-3527 | **逐字拷贝**（`split_if_not_byte_contiguous` 字节寻址用，删） |
| `HIXLConnector(KVConnectorBase_V1, SupportsHMA)` | :1411-1526 | **逐字拷贝**（dispatcher，仅改类名 + 内部 scheduler/worker 类名） |
| `HIXLConnectorScheduler` | :1529-1863 | **几乎全拷**（保留 `get_num_new_matched_tokens`/`update_state_after_alloc`/`build_connector_meta`/`request_finished(_all_groups)`/`set_xfer_handshake_metadata*` 等全部方法，含 PP/CP 分支） |
| `KVCacheSendingThread`（P 端，只回握手） | :244-406 | **逐字拷贝**，仅把返回的 `MooncakeAgentMetadata` → `HixlAgentMetadata` |
| `KVCacheRecvingThread`（D 端） | :408-1378 | 拷贝骨架（队列/socket pool/`_send_done_recv_signal`），**重写 `_get_remote_metadata` + `_transfer_kv_cache_all_groups`**（见 §3.3/3.4） |
| `HIXLConnectorWorker` | :1866-3417 | 拷贝几何逻辑，**重写 `__init__` engine 句柄 + `register_kv_caches` + `start_load_kv`**（见 §3.1/3.2） |

> **Phase 1 实际差异**：上表"逐字拷贝/几乎全拷"是完整 V1 目标。Phase 1 最小集实际**砍掉**：Scheduler 的 Mamba/SWA/compress/truncate 路径、RecvingThread 的 reformat 与字节簿记、Worker 的 `_get_kv_split_metadata` CP 分支/HMA rank 选择/Mamba helpers、`split_if_not_byte_contiguous`。完整清单见 §6.3。

---

## 2. HixlAgentMetadata 完整定义

替换 mooncake 的 `MooncakeAgentMetadata`。**去字节寻址字段，加 HIXL 路由字段**：

```
engine_id: str
cluster_id: int            # 本 rank 的 LLMDataDist cluster_id（D 用它作 pull 的 BlocksCacheKey.cluster_id）
listen_ip: str             # llm listen ip（D→P link 用）
listen_port: int           # llm listen port
model_id: int = 0          # BlocksCacheKey.model_id
num_tensors_per_group: list[int]   # 每 group 的 Cache.num_tensors（=组内层数×2）
# 以下保留（reformat/block 展开仍需）
kv_group2layeridx: dict
block_size: int
num_blocks: int            # 每 tensor 注册的 block 数（CacheDesc.shape[0]）
block_size_scale: list[list[int]]   # logical/tensor block 比例（block 展开需要）
local_ip: str = ""         # 多节点 host 映射
handshake_port: int = 0    # ZMQ 端口
```

**删掉**：`te_rpc_port`、`kv_caches_base_addr`、`block_lens`、`block_strides`（HIXL 按 cluster_id 路由、按 `CacheDesc.shape` 算地址，不做裸字节算术）。

---

## 3. Worker 改造细节（4 个改造点）

### 3.1 `__init__`（engine 句柄）— 改写 mooncake :1931-1935

```
cluster_id, listen_ip, listen_port = self._compute_identity()   # 见 §4
self.hixl = get_datadist(
    kv_role=self.kv_role, cluster_id=cluster_id,
    listen_ip=listen_ip, listen_port=listen_port,
    device_id=current_npu_device_id(),
    link_timeout_ms=extra["link_timeout_ms"],
    extra_options=extra,
)
self.cache_manager = self.hixl.cache_manager
self.group_caches: dict[int, Cache] = {}     # kv_cache_group_id → 注册返回的 Cache
self.model_id = extra.get("model_id", 0)
```
**去**：`self.engine = global_te...`、`self.te_rpc_port = ...`、`self.kv_caches_base_addr`、`self.remote_te_port`、`self.remote_block_stride_per_addr`（字节寻址簿记）。

### 3.2 `register_kv_caches` — 改写 mooncake :2224-2315

保留 `kv_group2layeridx` / `block_size` / `block_size_scale` 的计算（block 展开还要用；reformat 走 `self.kv_caches` tensor 自身 shape，无需字节 stride），**替换注册调用**：

```
from llm_datadist import CacheDesc, BlocksCacheKey, Placement, DataType
for group_id, (group_spec, layer_indices) in self.kv_group2layeridx.items():
    num_tensors = len(layer_indices) * 2          # K+V 每层（register 无 num_tensors 上限）
    addrs = [int(t.data_ptr()) for t in kv_tensors_of_group(group_id)]  # 断言 contiguous + 同 shape
    cache_desc = CacheDesc(
        num_tensors=num_tensors,
        shape=[num_blocks, block_size, *per_head_shape],   # 与 mooncake block 几何一致（含 num_blocks 维）
        data_type=_torch_dtype_to_llm_dtype(kv_dtype),
        placement=Placement.DEVICE,
    )
    cache = self.cache_manager.register_blocks_cache(
        cache_desc, addrs,
        BlocksCacheKey(self.hixl.cluster_id, self.model_id),
        remote_accessible=(self.kv_role == "kv_producer"),   # P 暴露，D 本地
    )
    self.group_caches[kv_cache_group_id] = cache
# 组装 HixlAgentMetadata（含 cluster_id/listen_*/num_tensors_per_group）交 SendingThread
```
- P/D 都 register：P 端 `remote_accessible=True`（被远端读），D 端 `False`（本地 `dst_cache`，供 §3.4 `pull_blocks` 写入）。
- **关键时序**：`register_blocks_cache` **必须在 link 之前**（否则 `raise_if_true(_is_call_linked and remote_accessible)` 报错）。已由 `__init__` 时 register 保证。
- `_torch_dtype_to_llm_dtype`：bf16→`DT_BF16`、fp16→`DT_FLOAT16`、fp32→`DT_FLOAT`、int8→`DT_INT8`（`DataType` 在 `data_type.py`，非 `llm_types.py`）。

### 3.3 `_get_remote_metadata`（D 端握手解码）— 改写 mooncake :1274-1306

从 `HixlAgentMetadata` 解出：`remote_cluster_id`、`remote_listen_ip`、`remote_listen_port`、`num_tensors_per_group`（替代 mooncake 的 `remote_te_port`/`kv_caches_base_addr`）。解码后**立即 `ensure_linked`**（D3）：

```
self.hixl.ensure_linked(
    remote_cluster_id=meta.cluster_id,
    remote_ip=meta.listen_ip,
    remote_port=meta.listen_port,
)
```
缓存：`self.remote_cluster_id[engine_id][handshake_port] = meta.cluster_id`。

### 3.4 `_transfer_kv_cache_all_groups`（核心改写）— 替换 mooncake :731-903

**保留** :731-795 的拆解（`group_pulls` 遍历、`pp_layer_indices` 过滤、`group_concurrent_contiguous` 分组 → 产出 kernel 粒度 `grouped_remote_block_ids` / `grouped_local_block_ids`）。

**删除** :762-884 的 `src_list/dst_list/length_list` 字节算术 + :896 `batch_transfer_sync_read`。

**替换**为按 (group, 远端 P cluster) 调 `pull_blocks`：

```
for group_pull in group_pulls:
    group_id = group_pull.group_id
    remote_cluster_id = self.remote_cluster_id[remote_engine_id][remote_handshake_port]
    dst_cache = self.group_caches[kv_cache_group_id]
    for chunk_remote, chunk_local in zip(grouped_remote_block_ids, grouped_local_block_ids):
        self.cache_manager.pull_blocks(
            BlocksCacheKey(remote_cluster_id, self.model_id),
            dst_cache,
            src_blocks=chunk_remote,       # P 的 kernel block id 列表（现成）
            dst_blocks=chunk_local,        # D 的 kernel block id 列表（现成）
            src_layer_range=layer_range,   # PP 过滤后的层范围（PP=1 时传 None）
            dst_layer_range=layer_range,
            tensor_num_per_layer=2,
        )
```
- **后置 reformat**（mooncake :916-982）**原样 fork、无需改动**（操作 `self.kv_caches` tensor + block_ids，引擎无关）。**触发条件**：仅 `num_group_pulls>1`（TP>1，见 §5）或 `enable_kv_nz` 时执行（:960-963、:989）；**Phase 1（TP=1、默认非 NZ）下为 no-op**。
- **错误处理**：mooncake 查 `ret<0`；HIXL `pull_blocks` 抛 `LLMException`。在 `_transfer` 外层 try/except 捕获 `LLMException`，映射到 `_mark_failed_recv_request`（mooncake :587），让 `get_block_ids_with_load_errors` 上报失败 block，scheduler 触发重算。
- **TP 分支**：`tp_num_need_pulls==1` 直接 pull 到 D 真实 block；`>1` 走 §5 staging 方案。

### 3.5 `start_load_kv`（Phase 1 简化）

mooncake 的 `start_load_kv` 调 `_get_kv_split_metadata`（复杂 CP 端口映射）+ `_get_group_pulls_metadata`。Phase 1（TP=1/单 group/PP=1）简化为：单 P rank、单 shard、每 group 一个 `GroupPull(num_group_pulls=1)`，直接 `kv_recv_thread.add_request`。

---

## 4. cluster_id / listen_port 算式（每 rank 唯一）

```
device_index = (pp_rank*pcp_size + pcp_rank)*tp_size + tp_rank        # 复用 mooncake :297/:1927
cluster_id   = cluster_id_base + dp_rank*(tp*pp*pcp) + device_index
listen_port  = listen_port_base + dp_rank*(tp*pp*pcp) + device_index   # 可复用 handshake_port
```
- P/D 用不相交 `cluster_id_base`（extra_config 必填，如 P=1000、D=2000）。
- `listen_ip` 取 `side_channel_host`（同 mooncake 的 `get_ip()`）。
- Phase 1（pcp=1）退化为 `device_index = tp_rank`，`cluster_id = cluster_id_base + dp_rank*tp + tp_rank`。

---

## 5. TP>1 的 head shard 拼接（staging 方案，Phase 2）

mooncake V1 靠字节寻址把 N 个 P rank 的 head shard **写到 D 真实 block 的 split i 位置**，写完后 D block 布局为 `[block, split, token, head_per_split, dim]`，再后置 `reformat` 做 `transpose(split, token)`（mooncake :966-1011）。HIXL `pull_blocks` 只能整块写、写不进 split，故需 staging：

```
对每个 tp_num_need_pulls>1 的 group，D 端额外分配一个 staging Cache：
  shape = [num_logical_blocks * tp_n, block_size, head_per_split, dim]
          （head_per_split = 该 group 一个 P rank 的 head 数，取自 P 注册的 block_shape_per_addr）
  register_blocks_cache(..., remote_accessible=False)   # D 本地接收缓冲

传输（替代 mooncake 的字节偏移写入）：
  for i, P_rank in enumerate(该请求要拉的 P ranks):         # i = tp_offset
      dst_blocks = [b * tp_n + i  for b in logical_blocks]    # shard i → staging block b*tp_n+i
      pull_blocks(BlocksCacheKey(P_cluster_of_rank_i, model_id),
                  dst_cache=staging, src_blocks=logical_blocks, dst_blocks=dst_blocks,
                  src_layer_range=pp_layer_range, tensor_num_per_layer=2)

后置 reformat（新写，借鉴 mooncake transpose 逻辑，但从 staging 读、写 D 真实 cache）：
  staging [num_blocks*tp_n, block_size, head_per_split, dim]
    → view [num_blocks, tp_n, block_size, head_per_split, dim]
    → transpose(1,2) → [num_blocks, block_size, tp_n, head_per_split, dim]
    → 散列写到 D 真实 cache 对应 block（index_copy_）
```

要点：
- staging buffer 按 group 分配，pull+reformat 完成后复用（常驻内存 = 活跃 block 数 × tp_n × P-shard-block-size）。
- reformat 是**新代码**（数据源是独立 staging buffer，非 mooncake 的 in-place），transpose 核心逻辑照搬 `reformat_kv_cache_hybrid_linear_torch`（mooncake :985-1011）。
- 其余并行维度（PP/PCP/DCP/HMA/Mamba）从 V1 fork，传输映射到 `pull_blocks`，无需 staging。

---

## 6. Phase 1 实现状态（2026-07-17）

Phase 1 最小集已实现：`hixl_connector.py`（~780 行，`ast.parse` 语法自查通过）+ 注册到 `__init__.py`（`HIXLConnectorV1`）。

### 6.1 已落实（对照 §3 改造点）
- **Worker.__init__**：`get_datadist` + `_compute_identity`（§4 算式，Phase 1 `pcp_size=1`）。
- **register_kv_caches**：每 group 一个 `register_blocks_cache`（P 端 `remote_accessible=True`，D 端 `False`）；K/V shape 一致 + `block_size_scale==1` assert。
- **_get_remote_metadata**：解码 `HixlAgentMetadata` → 立即 `ensure_linked`（D3）→ 存 `remote_cluster_id`。
- **_transfer**：`pull_blocks(BlocksCacheKey(remote_cluster_id, model_id), dst_cache, src/dst_blocks)`，`group_concurrent_contiguous` 分块；`LLMException` → `_mark_failed_recv_request`。
- **shutdown**：`shutdown_datadist`（unlink + finalize）。
- **ZMQ 控制面**：`KVCacheSendingThread`（ROUTER）+ `KVCacheRecvingThread`（REQ，socket 池）骨架 fork。

### 6.2 Phase 1 assert 约束（超出即清晰报错）
`tp_size==1`、`pp_size==1`、`block_size_scale==1`（标准 FullAttention，非 compress/SWA）、`num_group_pulls==1`、同 group K/V tensor shape 一致。

### 6.3 砍掉的（Phase 2/3 待补）
TP>1 staging（§5）、reformat（TP=1+非NZ 下 no-op，未实现）、Mamba（`_append_mamba_transfer_meta` 等）、PCP/DCP（端口映射 + `_get_kv_split_metadata` CP 分支）、HMA 多 group（`_get_hybrid_remote_rank_group_pulls` 等）、SWA/compress/MTP、`split_if_not_byte_contiguous`（字节寻址 helper）。

### 6.4 待 NPU 验证（代码无法定论）
1. **冒烟**：P `register_blocks_cache` → D `ensure_linked` → D `pull_blocks` 单 block，断言字节与 P 一致。
2. **端到端**：`--kv-transfer-config kv_connector=HIXLConnectorV1`，TP=1/PP=1/单 FullAttention 模型，P/D 双进程，与 `MooncakeConnectorV1` 同配置 KV 逐位对齐。
3. **HIXL 独有路径**：多 tensor 独立 `register_blocks_cache`（mooncake 走合并 region，未验过）、`pull_blocks` 整块写入的字节布局。

---

## 7. 实施分期（目标：完整 V1；建议分阶段降低风险）

最终交付对标 `MooncakeConnectorV1` 全部能力（~1500-2000 行），分三期推进，每期可独立验证：

- **Phase 1（主链路）** ✅ 已实现（最小集）：fork 类骨架 + ZMQ 控制面 + `__init__`/`register_kv_caches`/`_get_remote_metadata`/`_transfer` 四个改造点；TP=1、单 FullAttention group、PP=1。reformat fork（TP=1+非 NZ 下 no-op）、staging 不做。
- **Phase 2（TP staging + HMA）**：补 §5 staging + reformat，支持 `tp_num_need_pulls>1`；多 group + `request_finished_all_groups`。
- **Phase 3（PP/PCP/DCP + Mamba）**：fork `_get_kv_split_metadata` CP 分支、`_append_mamba_transfer_meta` 等，全并行维度覆盖。

**验收**：每期都以"与 `MooncakeConnectorV1` 同 P/D 几何输出逐位对齐"为标准。
