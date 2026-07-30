# HIXL Connector Phase 3 实施计划

> **PP / PCP / DCP + Mamba**（主线）；**+ MTP / Eagle 草稿层 KV 转移**（独立先落地子项，见设计文档 [§3.9](./hixl-connector-design.md#39-phase-3-子项mtp--eagle-草稿层-kv-转移设计)）。
> 配合 [`hixl-connector-design.md`](./hixl-connector-design.md)（背景+设计）、[`hixl-connector-implementation.md`](./hixl-connector-implementation.md)（编码细节）、[`hixl-connector-phase2-plan.md`](./hixl-connector-phase2-plan.md)（上一期）使用。
> 生成日期：2026-07-28
> 代码引用用 `文件:行`（相对仓库根）。`hixl_connector.py` = [`vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py)；`mooncake_connector.py` = [`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)。行号为 `vllm-ascend-v0.23.0` 当前状态，实施前需复核。

---

## 0. 实施后状态（2026-07-28，Phase 3 已落地）

> 本计划为实施前文档。Phase 3 A/B/C/D 四子项 + 卡点修复均已落地到 `hixl_connector.py`（~2785 行）。下方各节为"计划"语气（"未实现/待补/骨架"），**已过时**；以本节为准。行号均为 Phase 2 旧值，已偏移。

- **§1.1 口子表**（11 项）：10 项已放开/实现（`assert pp_size==1`、`assert not cp_transfer`、pcp 硬编码、`num_attn_module=1`、`_transfer` 全量 layer_range、request_finished 无裁剪、Mamba 骨架、无 need_truncate/truncate/state_prefill、无 speculative_config 均已解决）；仅 `assert scale==1` 保留（MLA/compress 专项，符合 §6）。
- **§4 实施清单 15 步**：步 1-4、6、8-14 已做；步 5、7 为 **B3 偏离**（见下）；步 15 验收待 NPU。
- **B3 偏离**（步 5、7、§3.5、§5.2 步3、§1.3 认知1）：`_transfer` 未用 `range(pp_first,pp_end)`，保留 `range(num_layers)`（行 806）。因每 PP rank 注册 cache 仅含本段层（vLLM PP 隔离 `PPMissingLayer`+`_project_kv_cache_groups_to_worker`），pull 全部本段层即等价。`pp_layer_indices` 建而未用。经 agent 核实 vLLM PP 下 kv_caches 确按 rank 隔离，本判断正确，计划 §3.5 `range(pp_first,pp_end)` 为误。
- **r_blk>1 assert**（计划未列）：`_get_kv_split_metadata_cp` 加 `assert r_blk==1 or use_mla`（行 2185）。r_blk>1（Bd>Bp）需 MLA/compress（scale>1 使 kernel_size 整除 Bp），mooncake scale=1 下亦不支持（`_local_kernel_ids_for_shard` 的 `kernels_per_p_block=0` 产出空）。归 MLA 专项。
- **Mamba align final state block**（§5.3 未列）：`_get_kernel_block_ids` Mamba 分支（行 2367-2384）align 模式只拉 final resident state block（remote 索引 `len-num_speculative_tokens-1`，local 索引 0，fork mooncake :867-869）。
- **Mamba state 多 P-rank**（§5.3 步3 风险已缓解）：`_transfer` 加 `is_state_group` 判定（行 764），`tp_n>1 and not is_state_group` 才走 staging（行 766），state 直落真实 cache。且 vLLM connector 侧 hetero-TP 仅支持 `tp_ratio==1`（base_worker.py:1761），Qwen3.6 P/D 同 TP → 不需 state staging。
- **`_append_mamba_transfer_meta`**（§5.3 步4）：显式 drop（行 1809-1810），block 寻址无对应物。
- **剩余缺口**（MLA/compress、SWA、sparse、NZ、SFA、conv_padding）：见 [`hixl-connector-gap-design.md`](./hixl-connector-gap-design.md)。

---

## 1. 范围与现状

Phase 3 目标（设计文档 [§3.8](./hixl-connector-design.md#38-实施分期与-phase-1-状态)）：

- **PP（Pipeline Parallel）**：`pp_size>1`，每个 PP rank 只拉/发自己负责的层段。
- **PCP / DCP（Context Parallel）**：prompt 跨多个 P worker 分片，D 按头部几何从多个 CP rank 拉。
- **Mamba / state group**：混合架构中 Mamba state 不按 head 分片、不裁推测 block，需独立 group 处理。
- **MTP / Eagle 草稿层**（子项，可先落地）：draft 层 KV 随主链路转移，仅层范围扩展 + 推测 block 裁剪。

### 1.1 Phase 2 留的口子（拦截点）

| 拦截 | 位置 | 含义 |
|---|---|---|
| `assert self.pp_size == 1` | `hixl_connector.py:1222` | PP>1 未支持（Phase 3 放开） |
| `self.pcp_size = 1` 硬编码 | `hixl_connector.py:264` | PCP 未接入（worker 侧无 `get_pcp_group()`） |
| `device_index = (pp_rank*1 + 0)*tp + tp_rank` | `hixl_connector.py:1292` | pcp 项硬编码为 0（无 `pcp_rank`） |
| `assert not cp_transfer` | `hixl_connector.py:1604-1605` | `_get_group_pulls_metadata` 的 CP 分支禁用 |
| `assert scale == 1` | `hixl_connector.py:1881` | MLA/compress 的 kernel-block 展开未做（见 §5.4 注） |
| `num_attn_module = 1` 硬编码 ×2 | `hixl_connector.py:792, :1332` | longcat_flash 双 attn module 静默错配（MTP 子项一并修） |
| `_transfer` 全量 layer_range | `hixl_connector.py:748,756-757` | `src_layer_range=range(num_layers)` 未按 PP rank 过滤 |
| `request_finished` 无裁剪 | `hixl_connector.py:1132-1165` | 直接用 `block_ids`，无 `_get_transfer_block_ids`（MTP 裁剪 + state group 不裁） |
| Mamba 分支仅骨架 | `hixl_connector.py:1549-1571` | `_get_hybrid_remote_rank_group_pulls` 的 MambaSpec 分支 assert + "kept for parity"，未真做 state group register/transfer |
| 无 `need_truncate`/`_truncate_request_for_prefill`/`_state_prefill_token_count` | — | Mamba 末 token 重算（P 丢末 token、D 从 N-1 起）未实现 |
| 无 `speculative_config` 处理 | — | MTP/Eagle 草稿层未识别 |

### 1.2 已就绪、可复用的骨架

- `HIXLConnector(KVConnectorBase_V1, SupportsHMA)`（`:766`）已继承 `SupportsHMA`；`request_finished_all_groups`（`:1003`）已转发，scheduler 侧 `request_finished`（`:1132`）按多 group 循环骨架已具备——**多 group 释放本就通**。
- `register_kv_caches`（`:1813`）已按 group 循环（`:1852`），一个 group 一个 `register_blocks_cache`（`:1899`），`num_tensors_per_group` 已收集（`:1875`）。
- ZMQ 控制面（`KVCacheSendingThread` / `KVCacheRecvingThread`）沿用，与引擎无关；`HixlAgentMetadata`（`:105`）已去字节寻址字段。
- **block 几何已 fork（Phase 2 落地）**：`_get_kv_split_metadata`（`:1654`，No-CP 分支）、`_get_group_pulls_metadata`（`:1594`）、`make_group_pulls`（`:1617`）、`_get_attention_group_num_need_pulls`（`:1393`）、`_get_attention_group_remote_rank`（`:1425`）、`_get_remote_ranks_for_req`（`:1481`，**已含 `prefill_pp_size` 维**：`:1505,1526,1530`）、`_get_kernel_block_ids`（被 `:1677` 调用）。
- **staging + reformat 已落地（Phase 2）**：`_stash_pending_reformat`（`:802`）、`_reformat_pending_kv_caches`（`:811`）、`_apply_kv_cache_reformat`（`:821`）、`staging_caches`（`:1917`）、`_init_staging_caches`（`:1918`）—— **TP>1 head 拼接路径已通，Mamba/PP/PCP 复用**。
- `GroupPull` 已含 `prefill_pp_rank` 字段（`:1567`，Phase 2 已加）—— **PP rank 透传链路已通**。
- `_get_group_pulls_metadata` 签名已含 `remote_pcp_size`/`remote_dcp_size` 形参（`:1600-1601`）—— **CP 入口已预留**。

### 1.3 HIXL 相对 mooncake 缺失的关键件

| 件 | mooncake 锚点 | HIXL 状态 |
|---|---|---|
| `pp_layer_indices` 建立（每 PP rank 的 `[first,end)` 层段） | `:529-532`（`get_prefill_pp_indices`） | **无**（`_prefill_pp_size` 已读 `:1376`，但未建层段表） |
| `_transfer` 按 PP rank 过滤层 | `:820-824` predicate + `:830` 取 `group_pull.prefill_pp_rank` | **全量 layer_range**（`:756-757`） |
| `pcp_size`/`pcp_rank`/`dcp_size`/`dcp_rank`（worker） | `:1999-2005` | **硬编码 pcp=1**（`:264`），无 dcp |
| CP 几何（`_get_kv_split_metadata` CP 分支） | `:2694` 之后的 `:2720-2769`（`context_parallel_parameters_check`/`get_kv_head_groups`/`get_cp_group_meta`）+ `:2641-2643`（`local_cp_rank/size`、`remote_cp_size`） | **assert 禁用**（`:1604`） |
| `is_state_group` 判定 + `group_transfer_info` | `:143`(字段)、`:1684,1696`(scheduler 建)、`:1710-1749`(消费) | **无** `group_transfer_info`，`_get_hybrid_remote_rank_group_pulls` Mamba 分支仅骨架（`:1549-1571`） |
| `_get_transfer_block_ids`（裁推测 block + state group 不裁） | `:1710-1733` | **无**（`request_finished` 直接用 `block_ids`，`:1155`） |
| `_state_prefill_token_count`（D 侧 N-1）/ `_truncate_request_for_prefill`（P 侧丢末 token） | `:1751-1756` / `:1758` | **无**（Mamba 末 token 重算未做） |
| `_append_mamba_transfer_meta`（Mamba 状态转移元数据） | `:1111` | **无** |
| `_get_mamba_conv_padding` / `_get_registered_kv_tensor_buffers` 的 conv_padding 对齐 | `:2267,2274-2293`（`base_addr -= conv_padding`） | **不适用**（block 寻址无裸字节基址，见 §5.4） |
| `num_attn_module`（longcat_flash=2） | `:1200,2183` | **硬编码 1**（`:792,1332`） |
| `num_draft_layers`（MTP/Eagle 识别） | `:539-551` | **无** |
| 末 PP rank `end_layer_index += num_draft_layers` | `:822-824` | **无**（PP layer 过滤未做，更无 draft 扩展） |

> **关键认知（⚠ 与 mooncake 不可照搬）**：
> 1. **PP 层过滤用 `pull_blocks` 原生 layer_range，不用 predicate**。mooncake 在 `_transfer` 内用 `pp_layer_indices` predicate 过滤层索引集合再字节寻址（`:820-830`）；HIXL 是 block 索引寻址，`pull_blocks` 原生支持 `src_layer_range`/`dst_layer_range`（设计文档 [§3.6.1](./hixl-connector-design.md#361-数据面kv-传输齐-全) "PP 按 layer 过滤 ✓ 原生"）。故 **`_transfer` 只需把 `range(num_layers)` 换成 `range(pp_first, pp_end)`**（末 rank 含 draft 层），**无需** fork mooncake 的 predicate 层过滤。
> 2. **state group 不 head-shard，不经 staging**。Mamba state 按 `num_group_pulls = prefill_tp/decode_tp` 拉（mooncake `:839-841`、HIXL `:1549-1571` 骨架），每个 P rank 拉的是完整 state（非 head shard），**直接落 D 真实 cache，不走 staging/reformat**（reformat 仅 attention group 的 `tp_n>1` 触发，`:770`）。
> 3. **conv_padding trick 不适用**。mooncake `:2293` 的 `base_addr -= conv_padding` 是字节寻址下的基址对齐；HIXL block 寻址无裸字节基址，**整体删除**——若 draft/Mamba 层与 attention 同 group 需对齐，走 block 级处理（§5.4），不在此 trick。
> 4. **MTP 裁剪是裁 block 数（整块），天然兼容 block 寻址**（设计文档 [§3.9.4](./hixl-connector-design.md#394-与已有-phase-23-机制的复用与适配)）——比 mooncake 字节 sub-range 裁剪更简单。
> 5. **`block_size_scale>1` 与 Mamba 无关**。`scale>1` 是 MLA/compress 把多个 logical block 打包进一个 tensor block 的比例（phase2-plan §4.1）；Mamba state group 是另一回事（不触发 scale）。Phase 3 若不含 MLA/compress，`scale==1` assert 可不放开（见 §5.4 注、§6 降级）。

---

## 2. 子项划分与依赖

Phase 3 拆为四个子项，按依赖与风险排序：

| 子项 | 依赖 | 可独立先落地？ | 风险 |
|---|---|---|---|
| **A. MTP/Eagle 草稿层** | 仅层范围扩展 + block 裁剪 | ✅ 是（PP=1 即可，设计文档 §3.9） | 低（修 `num_attn_module` 顺手修 longcat_flash） |
| **B. PP** | `pp_layer_indices` + `pull_blocks` layer_range | ✅ 是（与 PCP 互斥，mooncake `:2002` assert `not(pp>1 and pcp>1)`） | 中（层段边界、末 rank draft 扩展） |
| **C. Mamba / state group** | `is_state_group` + 末 token 重算 + state 不裁 | 部分（与 HMA 多 group 已部分骨架，但 state register/transfer 待补） | 中（state 不 head-shard、末 token 重算时序） |
| **D. PCP/DCP** | CP 几何 + 多 CP rank 端口/head 分组 | ❌ 需 B/C 骨架（device_index 含 pcp、`_get_kv_split_metadata` CP 分支） | 高（几何最复杂，mooncake `:2720-2769`） |

**依赖图**：

```
A (MTP/Eagle) ──独立──► 先落地（修 num_attn_module + 裁剪 + 末 rank draft 扩展）
B (PP)        ──独立──► 放开 :1222 + 建 pp_layer_indices + _transfer layer_range
C (Mamba)     ──依赖 HMA 多 group 骨架（已就绪）──► 补 is_state_group + 末 token 重算 + state register
D (PCP/DCP)   ──依赖 B 的 device_index 改造 + C 的 state group 判定──► CP 几何
B+C 交互      ──末 PP rank 的 draft 层扩展（A 的 §3.9 步5）需 B 的 layer_range 落地后
```

> **建议落地序**：A → B → C → D。A 与 B/C/D 无强依赖，先做 A 可独立验收；B 是 D 的几何基础（device_index 含 pcp 项）；C 的 state group 判定被 D 的 CP 分支复用。

---

## 3. 交互与关键信息流

Phase 3 的改造本质是"让现有交互链路带上 PP/PCP/DCP/Mamba/draft 几何信息"。下面逐环节列出 **Phase 3 增量字段**。

### 3.1 启动期握手：worker → 本地 scheduler → 框架汇总

| 字段 | 含义 | Phase 2 | Phase 3 增量 |
|---|---|---|---|
| `cluster_id`/`listen_*` | 本 rank 路由 | 按 tp_rank 列表（phase2） | **device_index 改含 pcp 项**（`:1292`），`cluster_id`/`listen_port` 偏移随之变 |
| `num_tensors_per_group` | 每 group `Cache.num_tensors` | 已按 group | **state group 的 num_tensors 按 Mamba state 语义**（非 K/V×2，见 §5.4） |
| `kv_group2layeridx` | group→层映射 | 已多 group | **补 mtp/eagle 层索引**（A 子项，`:1332` 的 `num_attn_module` + mtp 层段，fork mooncake `:2183,2195-2204`） |
| `block_size_scale` | logical/tensor block 比例 | `[1]` | **MLA/compress 才放开**（§5.4 注）；Mamba/PP/PCP 不触发 |

### 3.2 P scheduler → D scheduler（经 `kv_transfer_params`）

| 字段 | 含义 | Phase 3 增量 |
|---|---|---|
| `remote_pcp_size`/`remote_dcp_size` | P 的 CP size | **须透传**（fork mooncake `:1496-1497` 读、`:1919-1920` 回填）——当前 HIXL `request_finished`（`:1152-1165`）未回填，D 端 `_get_group_pulls_metadata` 拿不到 CP size |
| `remote_ptp_size` | P 的 TP size | 已传（`:1160`） |
| `remote_block_ids` | P 算出的 prompt block | **须经 `_get_transfer_block_ids` 裁推测 block + state group 不裁**（fork mooncake `:1710-1733`，A/C 子项） |
| `num_prompt_blocks`/`remote_block_size` | block 几何 | 不变 |

### 3.3 D scheduler → D worker（经 `SchedulerOutput.kv_connector_metadata`）

`ReqMeta`（`:128-141`）Phase 3 须补字段（复核当前是否已含）：

| 字段 | 含义 | Phase 3 增量 |
|---|---|---|
| `remote_pcp_size`/`remote_dcp_size` | P 的 CP size | **须加**（D 端 `_get_group_pulls_metadata` 形参已就绪 `:1600-1601`，但 `ReqMeta` 须透传） |
| `local_block_ids`/`remote_block_ids` | 按 group 分组 | state group 元素天然多一个（C 子项） |
| `prefill_pp_rank` | 经 `GroupPull` 透传 | 已就绪（`:1567`），**`_transfer` 须消费**（B 子项） |

### 3.4 D worker `start_load_kv` → `KVCacheRecvingThread.add_request`

| 字段 | Phase 2 | Phase 3 增量 |
|---|---|---|
| `group_pulls` | `make_group_pulls` 生成真实 `num_group_pulls`/`remote_tp_offset` | **CP 分支**：`_get_group_pulls_metadata` 放开 `:1604` assert，按 CP head 分组生成多 CP rank 的 group_pulls（fork mooncake `:2720-2769`）；**Mamba group**：走 `:1549-1571` 真实 state 分支（C 子项） |
| `remote_handshake_port` | 按 TP 几何算 | **CP 几何算**：端口含 `pcp_rank_offset`/`dcp_repeat_offset`/`kv_head_group_offset`（mooncake `:2764-2769`） |
| `all_task_done` | 仅最后 group_pull 置 True | CP 多 rank 时按 CP 分组收尾（D 子项） |

### 3.5 `pull_blocks` 调用（数据面，唯一真正传 KV）

- **签名**（`hixl_connector.py:751-758`）：
  ```
  cache_manager.pull_blocks(
      BlocksCacheKey(remote_cluster_id, model_id),
      dst_cache,                              # group_caches[gid] 或 staging
      src_blocks=chunk_remote, dst_blocks=chunk_local,
      src_layer_range=range(num_layers),      # ← Phase 3 改为 PP rank 层段
      dst_layer_range=range(num_layers),
  )
  ```
- **PP（B 子项）**：`src_layer_range`/`dst_layer_range` 从 `range(num_layers)` 改为 `range(pp_first, pp_end)`（末 rank `pp_end += num_draft_layers`，与 A 子项的 draft 扩展合流）。**这是 HIXL 相对 mooncake 的简化**——mooncake 用 predicate 过滤层集合再字节算术（`:820-830`），HIXL 直接传 layer_range。
- **Mamba（C 子项）**：state group 的 `dst_cache` 直接是 D 真实 cache（**不经 staging**，state 不 head-shard）；`num_group_pulls = prefill_tp/decode_tp`，每 P rank 拉完整 state（非 shard）。
- **PCP/DCP（D 子项）**：`src_blocks`/`dst_blocks` 按 CP head 分组落点；CP 下 prompt 跨多 P worker，`chosen_rank_list` 含多 CP rank（fork mooncake `:2720-2769`）。

### 3.6 RecvingThread → P SendingThread（ZMQ 控制面）

- **`GET_META_MSG`**：Phase 2 多 rank `ensure_linked` 已就绪。Phase 3 无新增（CP/PP 的 rank 经 `remote_port_send_num` 索引）。
- **`DONE_RECVING_MSG`**：P 端 `task_tracker` 按 `remote_port_send_num` 计数（phase2 已实现）—— CP 多 rank 时计数达标逻辑复用。

### 3.7 Phase 3 信息流增量总览

| 环节 | Phase 2 | Phase 3 增量 |
|---|---|---|
| `HixlAgentMetadata` device_index | tp_rank 维 | **含 pcp 项**（`:1292`） |
| `kv_transfer_params` | `remote_ptp_size` 已传 | **补 `remote_pcp_size`/`remote_dcp_size`** + 裁后 `remote_block_ids` |
| `ReqMeta` | — | **补 `remote_pcp_size`/`remote_dcp_size`** |
| `_get_group_pulls_metadata` | No-CP only（`:1604` assert） | **放开 CP 分支**（fork `:2720-2769`） |
| `_transfer` layer_range | `range(num_layers)` 全量 | **PP rank 层段**（末 rank +draft） |
| `request_finished` | 直接用 `block_ids` | **`_get_transfer_block_ids` 裁推测 + state 不裁** + **末 token 重算** |
| Mamba group | 骨架 assert | **真实 state 分支**（不 head-shard、不 staging） |

---

## 4. 实施清单（按依赖序）

| 步 | 子项 | 内容 | HIXL 落点 | MC 对标 |
|---|---|---|---|---|
| 1 | A | `__init__` 识别 spec 配置（`num_draft_layers`） | `:1218` 附近（worker）、scheduler `__init__` | `:539-551` |
| 2 | A | `_build_kv_group2layeridx` 补 mtp/eagle 层索引 + 修 `num_attn_module` | `:1313,1332` | `:2183,2195-2204` |
| 3 | A | `_get_group_kv_caches` 修 `num_attn_module` + mtp 归属 | `:788,792` | `:1196-1206` |
| 4 | A | `request_finished`（P）裁推测 block（attention group） | `:1132-1165` | `:1710-1733` |
| 5 | A+B | 末 PP rank `end_layer_index += num_draft_layers` | `pp_layer_indices`（新）+ `_transfer` layer_range | `:822-824` |
| 6 | B | 放开 `assert pp_size==1`（`:1222`）+ 建 `pp_layer_indices` | worker `:1218` 附近、新 `_build_pp_layer_indices` | `:529-532` |
| 7 | B | `_transfer` 用 PP rank 层段替换全量 layer_range | `:748,756-757` | `:820-830` |
| 8 | C | `is_state_group` 判定 + `group_transfer_info`（scheduler） | scheduler `request_finished` `:1132` 附近 | `:1684,1696` |
| 9 | C | `request_finished` state group 不裁 + Mamba 末 token 重算（P 丢末 token、D 从 N-1） | `:1132`、scheduler `get_num_new_matched_tokens` | `:1710-1733,1751-1762` |
| 10 | C | Mamba state group register（num_tensors 语义）+ `_get_hybrid_remote_rank_group_pulls` Mamba 真分支 | `:1549-1571`、`register_kv_caches` `:1852` | `:1111,2274` |
| 11 | D | worker 取 `pcp_size`/`pcp_rank`/`dcp_size`/`dcp_rank` + `device_index` 含 pcp | `:264,1292` | `:1999-2005,2035` |
| 12 | D | `ReqMeta` + `kv_transfer_params` 透传 `remote_pcp_size`/`remote_dcp_size` | `ReqMeta` `:128-141`、`request_finished` `:1152` | `:121-122,1496-1497,1919-1920` |
| 13 | D | `_get_group_pulls_metadata` 放开 CP assert + CP 几何（`get_cp_group_meta` 等） | `:1594-1652`（assert `:1604`） | `:2694-2769` |
| 14 | D | `_get_kv_split_metadata` CP 分支（多 CP rank 端口/block） | `:1654` | `:2662,2720-2769` |
| 15 | — | 验收：与 MooncakeConnectorV1 同 P/D 几何逐位对齐 | — | 设计文档 §3.8 |

---

## 5. 各步展开

### 5.1 子项 A：MTP / Eagle 草稿层（先落地）

按设计文档 [§3.9.3](./hixl-connector-design.md#393-hixl-适配设计block-寻址) 的 5 步，HIXL 落点已核实行号：

1. **`__init__` 识别 spec 配置**（步 1）：在 worker `__init__`（`:1218` 附近，`total_layers` 之后）和 scheduler `__init__` fork mooncake `:539-551`：算 `self.num_speculative_tokens` / `self.num_draft_layers`。MTP `method=="mtp"` → 1；eagle → `draft_model_config.hf_config.num_hidden_layers`；无 spec → 0。
2. **`_build_kv_group2layeridx` 补 mtp/eagle 层索引 + 修 `num_attn_module`**（步 2，`:1313`）：fork mooncake `:2183,2195-2204` 全段——含 `next_mtp_layer_idx = total_layers` 递增、eagle3 "层 id 已占用"判定（`assigned_indices`）、`longcat_flash` `num_attn_module=2`（`:1332` 改为按 `model_type` 取 2/1）。**此项同时修复 longcat_flash 静默错配**（即使不做 MTP 也该修，见 §7 风险）。
3. **`_get_group_kv_caches` 补 mtp 归属 + `num_attn_module`**（步 3，`:788-800`）：fork mooncake `:1200-1206` 的 `layer_in_group`——mtp 层走 `layer_idx >= num_layers`，非 mtp 走 `extract_layer_index`，`num_attn_module` 从硬编码 1 改为按 `model_type` 取 2/1（`:792`）。
4. **`request_finished`（P 侧）裁掉推测 block**（步 4，`:1132-1165`）：P 侧算出 prompt 占的 block 后，对 attention-like group 裁掉末尾 `num_speculative_tokens` 个推测 block。fork mooncake `_get_transfer_block_ids`（`:1710`）——**block 数裁剪，与 HIXL block 寻址天然兼容**（§1.3 认知 4）。state group（Mamba）不裁（步 8/9 合流）。
5. **拉取层范围纳入 draft 层**（步 5）：PP=1 时 draft 层本就在本 rank 的层集合内，由步 2/3 自动纳入 `kv_group2layeridx`，无需 `pp_layer_indices` 扩展。PP>1 时末 rank `end_layer_index += num_draft_layers`（与子项 B 的步 5 合流，随 PP 主线）。

> **A 子项验收**：见设计文档 [§3.9.6](./hixl-connector-design.md#396-验收)——MTP 配置下 P 侧裁剪后 block 数 = `prompt_blocks - num_speculative_tokens`，与 mooncake 同配置逐位对齐。

### 5.2 子项 B：PP（Pipeline Parallel）

1. **放开 `assert pp_size==1`**（`:1222`）+ assert `not(pp>1 and pcp>1)`（fork mooncake `:2002`，PP 与 PCP 互斥）。
2. **建 `pp_layer_indices`**（步 6）：在 worker `__init__`（`:1218` 附近）fork mooncake `:529-532`：
   ```
   from vllm... import get_prefill_pp_indices   # vLLM 自带
   self.pp_layer_indices = {
       rank: get_prefill_pp_indices(self.total_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
       for rank in range(self._prefill_pp_size)
   }
   ```
   `_prefill_pp_size` 已读（`:1376`）。
3. **`_transfer` 用 PP rank 层段**（步 7，`:748,756-757`）：当前 `num_layers = dst_cache.cache_desc.num_tensors // 2`（全量），`src_layer_range=range(num_layers)`。改为按 `group_pull.prefill_pp_rank`（`GroupPull` 已含此字段，`:1567`）取层段：
   ```
   pp_first, pp_end = self.pp_layer_indices[group_pull.prefill_pp_rank]
   if spec_config is not None and group_pull.prefill_pp_rank == self._prefill_pp_size - 1:
       pp_end += self.num_draft_layers   # 末 rank 含 draft 层（与 A 步 5 合流）
   src_layer_range = range(pp_first, pp_end)
   dst_layer_range = range(pp_first, pp_end)
   ```
   > **这是 HIXL 相对 mooncake 的简化**（§1.3 认知 1）：mooncake 用 predicate 过滤层集合再字节算术（`:820-830`），HIXL 直接传 layer_range。**无需** fork predicate。
4. **P 侧 `request_finished`**：`remote_block_ids` 仍是 P 算出的 prompt block（PP 各 rank 各自的 block），无需改——PP rank 的层归属在 D 侧 `_transfer` 按 `prefill_pp_rank` 过滤层。

> **B 子项验收**：PP>1 配置下，每个 D PP rank 只拉自己层段的 KV，与 mooncake 同配置逐位对齐。

### 5.3 子项 C：Mamba / state group

1. **`is_state_group` 判定 + `group_transfer_info`**（步 8）：在 scheduler 侧 fork mooncake `:1684,1696`——为每 group 建 `GroupTransferInfo(is_state_group = any(isinstance(spec, MambaSpec) for spec in specs), ...)`。`need_truncate = use_compress or any(is_state_group)`（`:1673`）。
2. **`request_finished` state group 不裁 + 末 token 重算**（步 9，`:1132`）：
   - P 侧 `_get_transfer_block_ids`（fork `:1710-1733`）：state group `is_state_group=True` → `transfer_block_ids.append(blocks)`（不裁，`:1725`）；attention group → `blocks[:num_prompt_blocks]`（裁，`:1731-1732`，与 A 步 4 的推测 block 裁剪合流）。
   - **Mamba 末 token 重算**：P 侧 `_truncate_request_for_prefill`（fork `:1758`，丢末 prompt token，P 算 h(N-1) 而非 h(N)）；D 侧 `_state_prefill_token_count`（fork `:1751-1756`，从 N-1 起，重算末 token 得 h(N)）。`get_num_new_matched_tokens`（scheduler）对 Mamba 截断末 token。
3. **Mamba state group register + `_get_hybrid_remote_rank_group_pulls` Mamba 真分支**（步 10）：
   - `register_kv_caches`（`:1852`）：state group 的 `num_tensors` 语义——Mamba state 是单 tensor/层（非 K/V×2），fork mooncake `:2274-2300` 的 `_get_registered_kv_tensor_buffers_hybrid`（**删 conv_padding 字节对齐**，§1.3 认知 3）。
   - `_get_hybrid_remote_rank_group_pulls`（`:1549-1571`）：当前 MambaSpec 分支已有骨架（`num_group_pulls = prefill_tp/decode_tp`，`:1555`），去掉 "kept for parity" assert，**真实生成 state group_pulls**。state 不 head-shard → `remote_tp_offset` 语义为 P rank 偏移（非 head split 偏移），**不经 staging**（`_transfer` 中 `tp_n>1` 才走 staging，state group 的 `num_group_pulls` 是 P rank 数而非 head split 数，须确保 state 走 `:733` 的 `else` 直接落真实 cache 分支——复核 `tp_n` 语义）。
4. **`_append_mamba_transfer_meta`**（fork `:1111`）：Mamba state 的转移元数据（若 state 须单独组装）。

> **C 子项关键风险**（见 §7）：state group 的 `num_group_pulls` 语义与 attention head-shard 的 `tp_n` 语义不同——attention 的 `tp_n` 是 head split 数（走 staging），Mamba state 的 `num_group_pulls` 是 P rank 数（不走 staging，每 rank 拉完整 state）。`_transfer` 的 `:717 if tp_n > 1` 分支须区分：**state group 即使 `num_group_pulls>1` 也走 `else` 直落真实 cache**（因为 state 不拼 head）。须在 `_transfer` 加 `is_state_group` 判定或确保 `GroupPull` 透传 group 类型。

### 5.4 子项 D：PCP / DCP

1. **worker 取 CP size + `device_index` 含 pcp**（步 11，`:264,1292`）：
   - `self.pcp_size = get_pcp_group().world_size`、`self.pcp_rank = get_pcp_group().rank_in_group if pcp_size>1 else 0`、`self.dcp_size = get_decode_context_model_parallel_world_size()`、`self.dcp_rank = get_decode_context_model_parallel_rank() if dcp_size>1 else 0`（fork `:1999-2005`）。
   - `device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank`（`:1292`，fork `:2035`）；`cluster_id`/`listen_port` 偏移随之含 pcp（`:1293`）。
2. **`ReqMeta` + `kv_transfer_params` 透传 CP size**（步 12）：
   - `ReqMeta`（`:128-141`）加 `remote_pcp_size`/`remote_dcp_size`（fork `:121-122`）。
   - P 侧 `request_finished`（`:1152-1165`）回填 `remote_pcp_size=self.pcp_size, remote_dcp_size=self.dcp_size`（fork `:1919-1920`）。
   - D 侧 `build_connector_meta` 从 `kv_transfer_params` 提取透传给 `ReqMeta`（fork `:1496-1497`）。
3. **`_get_group_pulls_metadata` 放开 CP assert + CP 几何**（步 13，`:1594-1652`，assert `:1604`）：
   - 去掉 `assert not cp_transfer`，fork mooncake `:2694` 之后的 CP 分支：`context_parallel_parameters_check`（`:2720-2726`）、`get_kv_head_groups`（`:2727-2746`）、`get_cp_group_meta`（`:2748-2769`）。
   - `local_cp_rank = self.dcp_rank + self.pcp_rank * self.dcp_size`、`local_cp_size = self.dcp_size * self.pcp_size`、`remote_cp_size = meta.remote_pcp_size * meta.remote_dcp_size`（fork `:2641-2643`）。
4. **`_get_kv_split_metadata` CP 分支**（步 14，`:1654`）：当前仅 No-CP 分支（`:1665-1685`）。CP 下 prompt 跨多 P worker，`chosen_rank_list` 含多 CP rank，`remote_handshake_port_list` 按 CP head 分组（fork `:2720-2769` 的 `get_cp_group_meta` 产出）。

> **D 子项 HIXL 适配**：mooncake CP 几何输出字节地址算术，HIXL 只需 block 索引——**删字节算术，保留 CP head 分组几何**（同 phase2-plan §4.2 的 No-CP 适配原则）。CP 多 rank 的端口/head 落点计算须在 `group_concurrent_contiguous`（`:1374`）合并前确定。

---

## 6. 降级路径

若 Phase 3 分步落地，按风险从低到高：

- **先做 A（MTP/Eagle）**：PP=1 即可，独立验收；顺手修 longcat_flash `num_attn_module`。
- **再做 B（PP）**：与 PCP 互斥（`not(pp>1 and pcp>1)`），PP>1 + TP>1 + HMA 可先验。
- **C（Mamba）部分降级**：若先只做"state group 不裁 + 末 token 重算"（步 8/9），不做 state 独立 register（步 10 state num_tensors 语义）——则 state group 仍按 attention group 的 K/V×2 register，**仅适用 Mamba state 与 attention 同 group 同 shape 的模型**；独立 spec 的 Mamba 须做步 10。
- **D（PCP/DCP）最后**：几何最复杂，建议 B/C 稳定后再做；可先只做 PCP（不含 DCP），降低 `get_cp_group_meta` 的 `dcp_repeat_num` 维度复杂度。
- **MLA/compress 专项**：`block_size_scale>1`（`:1881` assert）与 Mamba/PP/PCP **无关**（§1.3 认知 5），**从 Phase 3 主线剥离**，归 MLA/compress 专项（可能更后）。Phase 3 若仅 FullAttention+Mamba，`scale==1` assert 不放开。

---

## 7. 验收（设计文档 [§3.8](./hixl-connector-design.md#38-实施分期与-phase-1-状态)）

每子项与 `MooncakeConnectorV1` 同 P/D 几何**逐位对齐**。

1. **A（MTP/Eagle）**：`speculative_config={method:mtp/eagle}`，P 侧裁剪后 block 数 = `prompt_blocks - num_speculative_tokens`，`External prefix cache hit rate: 100.0%`，draft 层 KV D 侧与 P 侧逐字节一致（设计文档 [§3.9.6](./hixl-connector-design.md#396-验收)）。
2. **B（PP）**：`pp_size>1`，每个 D PP rank 只拉自己层段，与 mooncake 同配置 KV 逐位对齐。
3. **C（Mamba）**：混合架构（FullAttention+Mamba），state group 不裁、末 token 重算正确，D 侧 state KV 与 P 侧逐位对齐。
4. **D（PCP/DCP）**：`pcp_size>1`/`dcp_size>1`，prompt 跨多 P worker 分片拉取，与 mooncake 同配置逐位对齐。
5. **断言**：放开 Phase 2 留的口子（`:1222 pp`、`:1604 cp`、`:264 pcp`、`:1292 device_index`、`:792,1332 num_attn_module`）；`scale==1`（`:1881`）仅在不含 MLA/compress 时保持。

---

## 8. 风险点

| 风险 | 说明 | 缓解 |
|---|---|---|
| longcat_flash 静默错配 | `num_attn_module=1` 硬编码（`:792,1332`），longcat_flash 模型算错层索引而不报错 | A 步 2/3 一并修（即使不做 MTP 也该修） |
| state group `num_group_pulls` 语义混淆 | attention 的 `tp_n` 是 head split 数（走 staging），Mamba state 的 `num_group_pulls` 是 P rank 数（不走 staging）。`_transfer` 的 `:717 if tp_n>1` 若不区分，state 会被误送 staging | `_transfer` 加 `is_state_group` 判定或 `GroupPull` 透传 group 类型；state group 强制走 `:733 else` 直落真实 cache（§5.3 步 3） |
| Mamba 末 token 重算时序 | P 丢末 token 算 h(N-1)、D 从 N-1 起重算末 token 得 h(N)，时序须与 `get_num_new_matched_tokens`/`request_finished` 协同 | fork mooncake `:1751-1762` 全段，scheduler `get_num_new_matched_tokens` 对 Mamba 截断（§5.3 步 9） |
| PP 末 rank draft 层扩展 | `end_layer_index += num_draft_layers` 须在 PP layer_range 落地后（B 步 7）才能接入 A 步 5 | A 先做 PP=1（draft 层已在本 rank），PP>1 的末 rank 扩展随 B 合流 |
| CP 几何 `block_size_scale` 结构不匹配 | HIXL `block_size_scale[group]=[scale]` vs MC `block_size_scale[layer][cache]`（phase2-plan §4.2）；CP 分支若搬 MC `_get_kernel_block_ids`（`:2585`）会越界 | 移植 CP 前先统一结构或加 group→layer 索引转换（同 phase2-plan §4.2） |
| `cluster_id` 偏移碰撞 | `device_index` 含 pcp 项后，`cluster_id`/`listen_port` 偏移空间变化，P/D 不相交 base 须重算 | 复核 `cluster_id_base` 配置（P/D 不相交，设计文档 [§3.5.1](./hixl-connector-design.md#351-配置项kv_connector_extra_confighixl)） |
| PP 与 PCP 互斥 | mooncake `:2002` assert `not(pp>1 and pcp>1)`，PP>1 时 pcp 必须=1 | 步 6 放开 PP assert 时同步加互斥 assert（§5.2 步 1） |
| CP 多 rank link 时序 | `remote_accessible=True` 的 register 必须在 link 之前（设计文档 [§3.7](./hixl-connector-design.md#37-接口核对结论)） | P 被动不 link，D 端 link 顺序由 `_get_remote_metadata` 保证（phase2 已就绪，CP 多 rank 复用） |
| conv_padding 误搬 | mooncake `:2293` 的 `base_addr -= conv_padding` 是字节寻址 trick，直接搬会语义错误 | 整体删除（§1.3 认知 3）；block 级对齐走 §5.4 |

> 注：本计划基于设计文档与当前代码静态分析，未在 NPU 上实测。行号为 `vllm-ascend-v0.23.0` 当前状态（Phase 2 已完整落地），实施前需复核（fork 源 mooncake 行号仅作对照锚点）。MTP/Eagle 子项设计详见 [`hixl-connector-design.md` §3.9](./hixl-connector-design.md#39-phase-3-子项mtp--eagle-草稿层-kv-转移设计)。
