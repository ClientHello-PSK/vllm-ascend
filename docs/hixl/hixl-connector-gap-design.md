# HIXL Connector 缺失能力设计与实现流程

> 对照 `MooncakeConnectorV1`（[`mooncake_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)）的寻址无关能力缺口。
> 配合 [`hixl-connector-design.md`](./hixl-connector-design.md)、[`hixl-connector-phase3-plan.md`](./hixl-connector-phase3-plan.md) 使用。
> 生成日期：2026-07-28。代码引用 `文件:行`（相对仓库根）；mooncake 行号仅作 fork 锚点，实施前需复核。
> `hixl_connector.py` = [`vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py`](../../vllm_ascend/distributed/kv_transfer/kv_p2p/hixl_connector.py)。

---

## 1. 背景

HIXL Phase 3 已完成 A(MTP)/B(PP)/C(Mamba state)/D(PCP/DCP)，约束 `block_size_scale==1` 且 `r_blk==1`（P/D 同 block_size、无 MLA/compress）。相对 mooncake，HIXL 用 `pull_blocks` 替代了字节寻址（`batch_transfer_sync_read`/`kv_caches_base_addr`/`_append_mamba_transfer_meta` 字节算术等），这部分是寻址方式差异，**非缺口**。本文聚焦 mooncake 有、HIXL 无的**寻址无关能力缺口**（10 项），给出 block 寻址下的设计与实现流程。

---

## 2. 缺口总览

| # | 能力 | mooncake 锚点 | HIXL 状态 | 影响模型/场景 | 优先级 |
|---|---|---|---|---|---|
| G1 | MLA/compress `block_size_scale>1` | `:2374,2585,2618` | ❌ `assert scale==1`（hixl:2527） | DeepSeek-V3 MLA、DSV4 compress | 高 |
| G2 | `conv_padding`（Mamba+MTP conv state） | `:2267,2278-2293` | ❌ 全缺 | Mamba+MTP（Qwen3.6 潜在） | 高 |
| G3 | NZ layout（`enable_kv_nz`） | `:1059,1244-1366` | ❌ 仅 head 转置 | NPU NZ 排布 KV cache | 高 |
| G4 | SWA 滑窗裁剪 | `:1735,142,53` | ❌ 全缺 | 滑窗注意力（Mistral 等） | 中 |
| G5 | sparse attention（`use_sparse`） | `:2335,2862` | ⚠️ 硬编码 `False`（hixl:1497） | sparse/MoE-attn（`index_topk`） | 中 |
| G6 | SFA DCP replicate-K | `:3301,127,516` | ❌ 全缺 | SFA + DCP + prefix cache | 中 |
| G7 | `MLAAttentionSpec` 远端握手 | `:52,2166` | ❌ 未 import | MLA group spec 误判 | 中（G1 子项） |
| G8 | CP + MLA 联合 kernel（`r_blk>1`+`scale>1`） | `:2505,2618` | ❌ assert（hixl:2185） | CP + P/D block_size 不等 + MLA | 低（G1 子项） |
| G9 | `r_blk>1`（Bd>Bp，无 MLA） | `:2553-2566` | ❌ assert（hixl:2185） | P 细 block / D 粗 block 异构 | 低 |
| G10 | `enable_sfa_dcp_replicated_indexer` | `:516,916` | ❌ 缺 | SFA DCP | 低（G6 子项） |

**依赖关系**：G7、G8 是 G1 的子项；G10 是 G6 的子项。G2、G3 独立。G4、G5 独立。G9 依赖 G1（r_blk>1 需 scale>1 使 kernel_size 整除 Bp）。

---

## 3. 高优先级

### 3.1 G1：MLA / compress（`block_size_scale>1`）

**能力**：MLA（DeepSeek）把多个 logical block 压进一个 tensor block（`compress_ratio`），`block_size_scale = tensor_num_blocks / logical_num_blocks > 1`。传输需按 **kernel block**（`block_size/scale` 粒度）寻址，logical block 展开为 `scale` 个 kernel block。

**mooncake 锚点**：
- `block_size_scale[layer][cache]` 填充 `:2374-2379`
- `_expand_block_ids(bids, scale)` = `[bid*scale+off]`（`:2500`）
- `_get_kernel_block_ids`（`:2585-2616`）：`local_scale=block_size_scale[layer][0]`、`kernel_size=block_size/scale`、`remote_scale=remote_block_size/kernel_size`
- `_get_group_kernel_params`（`:2618-2633`）：返回 per-group `(local_scale, remote_scale, kernel_size)`
- `MLAAttentionSpec.storage_block_size = block_size//compress_ratio`（vllm `kv_cache_interface.py:402`）

**HIXL block 寻址适配**：
- HIXL `block_size_scale` 已是 **per-group `[scale]`**（`hixl:2504`），与 mooncake per-layer-per-cache `[layer][cache]` 结构不同。同 group 同 scale 时 per-group 足够；若同 group 内不同层 scale 不同（罕见），需升为 per-layer。
- `register_kv_caches`（`hixl:2499-2504`）：放开 `assert scale==1`，`scale = ref_shape[0] // self.num_blocks` 已算出，存 `block_size_scale.append([scale])`。
- `_get_kernel_block_ids`（`hixl:2348`）：`local_scale` 当前恒 1（`hixl:2387`），改为 `self.block_size_scale[group_idx][0]`；`_expand_block_ids` 已移植（`hixl:2338`），scale>1 时自动展开。
- `pull_blocks` 的 `src/dst_blocks` 用 kernel block id（展开后），layer_range 不变。

**实现步骤**：
1. 放开 `hixl:2527` 的 `assert scale==1`，改为 `assert scale >= 1`。
2. `_get_kernel_block_ids`（`hixl:2348`）：`local_scale = self.block_size_scale[group_idx][0]`（替换 `:2387` 硬编码 1 的来源）；验证 `kernel_size`、`remote_scale`、`remote_start_idx`（`:2369`）路径。
3. `block_size_scale` 结构复核：若需 per-layer，改 `hixl:2504` 为 `[[scale_per_layer...]]` 并调整索引。
4. G7：`import MLAAttentionSpec`；`_build_kv_group2layeridx` spec 序列化加 MLA 识别（`_get_spec_total_num_kv_heads` 把 MLA 当 `num_kv_heads==1`，fork `:2166`）。
5. 验收：DeepSeek-V3 MLA P/D 配置，与 mooncake 同配置 KV 逐位对齐。

**风险**：`block_size_scale` per-group vs per-layer 结构差异（§phase3-plan §8 已列）；CP 分支（`_get_kv_split_metadata_cp`）当前直接用 block id（G3 决策），MLA+CP 联合需 `_local_kernel_ids_for_shard`（G8）。

### 3.2 G2：conv_padding（Mamba+MTP conv state）

**能力**：MTP 草稿层若共享 Mamba conv state，mooncake 在 register 时 `base_addr -= conv_padding` 把 conv state buffer 纳入注册区一并传输（`:2278-2293`）。

**mooncake 锚点**：
- `_get_mamba_conv_padding`（`:2267-2272`）：`num_blocks * conv_shape.numel() * dtype_size`
- `_get_registered_kv_tensor_buffers`（`:2278-2293`）：`has_mtp → base_addr -= conv_padding`

**Qwen3.5 调查结论（2026-07-28，agent 核实）**：Qwen3.5 MTP 草稿层（[`qwen3_5_mtp.py:106-113`](../../vllm/vllm/model_executor/models/qwen3_5_mtp.py)）强制 `layer_type="full_attention"`，是**纯 attention 层**，无 GDN/Mamba 子模块，不共享 mamba conv_state。基础模型 GDN（`linear_attention`）层的 conv_state + ssm_state 已作为 MambaSpec group 的 tensor 被 HIXL C 子项 `register_kv_caches`（`_as_kv_cache_tuple` 展开 conv+ssm 加入 addrs，hixl:2478）注册并传输。**故 Qwen3.6 不受 G2 影响**。

**G2 实际影响场景**：仅当 MTP 草稿层本身是 mamba 层（共享基础模型 mamba conv state）时才需 conv_padding。当前 Qwen3.5 不属此场景，属罕见/未来模型。

**HIXL block 寻址适配**（若未来需补）：
- HIXL `register_blocks_cache` 按 `kv_caches[layer_name]` tensor 的 `data_ptr` 注册（hixl:2478-2479）。conv state buffer 不在 `kv_caches` 字典里 → 不被注册。
- **block 寻址下不能搬 `base_addr -= conv_padding`**（无裸字节基址）。改为：把 conv state 作为**额外 tensor** 加入 Mamba group 的 `addrs` 列表，`num_tensors` 相应增加，`CacheDesc.shape` 兼容。

**实现步骤**（未来需补时）：
1. 确认目标模型 MTP 草稿层是否共享 mamba conv state（Qwen3.5 已确认否）。
2. 若共享：`register_kv_caches` 的 Mamba group 收集 conv state buffer addr，加入 `addrs`；调整 `num_tensors` assert（hixl:2491）。
3. `_get_group_kv_caches` Mamba 分支返回含 conv state 的 tensor 集。
4. 验收：Mamba+MTP（mamba 草稿层）配置，D 侧草稿层 conv state 与 P 侧逐字节对齐。

**风险**：conv state 的 shape/block 结构与 K/V 不同，`CacheDesc` 需兼容异质 tensor（可能需独立 group 注册）。

### 3.3 G3：NZ layout（`enable_kv_nz`）

**能力**：NPU 上 KV cache 用 NZ 排布（16-末维），拉完后需 NZ scatter 写 D cache。

**mooncake 锚点**：
- `enable_kv_nz` 检测（`:1059` 调用处）
- `reformat_kv_cache`（`:1244-1319`）：`need_cat`/`need_nz` 分支
- `_cat_kv_cache`/`_nz_kv_cache`（`:1321-1366`）：`npu_paged_cache_load` + `_npu_reshape_and_cache` / `npu_scatter_pa_kv_cache`
- `reformat_kv_cache_hybrid_linear_torch`（`:1083-1092`）

**HIXL block 寻址适配**：
- HIXL `_reformat_staging_to_local`（`hixl:914`）仅做 TP head 转置（`transpose(1,2)`），无 cat/NZ/paged-load。
- NZ 下 D cache 是 NZ 排布，pull 落 staging 后需 `npu_scatter_pa_kv_cache` 写 NZ。fork mooncake `_nz_kv_cache`（`:1347` 附近），但数据源是 HIXL staging Cache（block 寻址），非字节 region。

**实现步骤**：
1. 加 `enable_kv_nz` 检测（读 vllm-ascend ascend_config 或 cache_config）。
2. `_reformat_staging_to_local` 加 NZ 分支：`if enable_kv_nz: npu_scatter_pa_kv_cache(...)` else 现有 head 转置。
3. `_get_group_kv_caches` 返回的 D cache tensor 需含 NZ 布局信息。
4. 验收：NZ 排布下 TP>1 P/D，D 侧 cache 布局与 mooncake 一致。

**风险**：NZ scatter 算子依赖 vllm-ascend CANN ops；staging Cache 的 tensor 布局需与 NZ scatter 输入匹配。

---

## 4. 中优先级

### 4.1 G4：SWA 滑窗裁剪

**mooncake 锚点**：`SlidingWindowSpec`（import `:53`）、`GroupTransferInfo.blocks_per_window`（`:142,1695`）、`_get_swa_transfer_block_ids`（`:1735-1749`，裁窗口尾 + 丢占位 block 0）、调用 `:1904`。

**HIXL 适配**：
1. `GroupTransferInfo` 加 `blocks_per_window` 字段（hixl:309 dataclass）。
2. `_get_group_transfer_info`（hixl:1180）读 `SlidingWindowSpec.sliding_window`，算 `blocks_per_window = cdiv(sliding_window, block_size)+1`。
3. `import SlidingWindowSpec`；加 `_get_swa_transfer_block_ids`（fork `:1735-1749`，寻址无关可直接 fork）。
4. `request_finished`（hixl:1132）调用链：`_get_transfer_block_ids` → `_get_swa_transfer_block_ids`（fork `:1904` 顺序）。

**验收**：SWA 模型 P/D，D 侧只收窗口内尾部 block，与 mooncake 一致。

### 4.2 G5：sparse attention

**mooncake 锚点**：`use_sparse` 检测 `index_topk`（`:2335`）；HMA 分支 `:2862,3559,3585` 的 sparse 特例（退化为单 head group，类似 MLA）。

**HIXL 适配**：
1. `use_sparse` 从 `False`（hixl:1497）改为 `_model_uses_sparse()`（检测 `hf_text_config.index_topk`）。
2. HMA 分支（`_get_remote_ranks_for_req` hixl:1481、`_get_cp_shard_pulls` 等）加 `use_sparse` 判定，sparse 下 rank 选择退化为单组（fork mooncake `:2728` 的 `if use_mla or use_sparse`）。

**验收**：sparse attention 模型 P/D，rank 选择与 mooncake 一致。

### 4.3 G6：SFA DCP replicate-K

**mooncake 锚点**：`_get_sfa_replicate_k_block_ids`（`:3301-3372`）、`ReqMeta.local_full_block_ids`（`:127`）、`enable_sfa_dcp_replicated_indexer`（`:516,916`）。

**HIXL 适配**：
1. `ReqMeta` 加 `local_full_block_ids` 字段（hixl:130 dataclass）。
2. 加 `_get_sfa_replicate_k_block_ids`（fork `:3301`，寻址无关）。
3. `enable_sfa_dcp_replicated_indexer` 检测（fork `:516`）。
4. CP 分支（`_get_kv_split_metadata_cp`）SFA 下按 global block 重建 K 副本。

**验收**：SFA + DCP + prefix cache，D 侧 K 块位置与 mooncake 一致。

---

## 5. 低优先级 / 边缘

### 5.1 G7：`MLAAttentionSpec` 远端握手（G1 子项）

`import MLAAttentionSpec`；`_build_kv_group2layeridx` 加 `_get_spec_total_num_kv_heads`/`_get_spec_num_key_value_heads`/`_get_kv_transfer_spec_key`/`_serialize_kv_group_spec`（fork `:2142,2166`）。MLA spec 当 `num_kv_heads==1` 处理。随 G1 一起做。

### 5.2 G8：CP + MLA 联合 kernel（G1 子项）

`_get_group_kernel_params`（fork `:2618`）+ `_local_kernel_ids_for_shard`（fork `:2505`）的 block 级改写。`r_blk>1` + `scale>1` 联合。放开 `hixl:2185` 的 `assert r_blk==1 or use_mla`。随 G1 + D4 一起做。

### 5.3 G9：`r_blk>1`（无 MLA）

依赖 G1（r_blk>1 需 scale>1 使 kernel_size 整除 Bp，mooncake `:2362` assert）。无 MLA 下 mooncake 也不支持（产出空）。**不单独实现**，随 G1。

### 5.4 G10：`enable_sfa_dcp_replicated_indexer`（G6 子项）

随 G6。

---

## 6. 优先级与落地序

| 批次 | 子项 | 依赖 | 说明 |
|---|---|---|---|
| 1 | G2（conv_padding） | 需先确认 Qwen3.6 MTP 是否共享 mamba conv state | Qwen3.6 潜在卡点，独立可做 |
| 2 | G3（NZ） | 独立 | NPU NZ 排布，影响所有 NZ 模型 |
| 3 | G1 + G7（MLA/compress + 握手） | 独立 | DeepSeek-V3，最大缺口 |
| 4 | G4（SWA） | 独立 | 滑窗模型 |
| 5 | G5（sparse） | 独立 | sparse 模型 |
| 6 | G6 + G10（SFA） | 独立 | SFA 场景 |
| 7 | G8 + G9（CP+MLA / r_blk>1） | 依赖 G1 | MLA + CP 联合，最后 |

**建议**：G2 先做（Qwen3.6 潜在卡点，且需先查 Qwen3.6 MTP 架构确认）；G3 次之（NZ 是 NPU 常见）；G1 是最大工程（MLA 全链路）。

---

## 7. 验收

每子项与 `MooncakeConnectorV1` 同 P/D 几何**逐位对齐**：
- G1：DeepSeek-V3 MLA，`block_size_scale>1`，KV 逐位对齐
- G2：Mamba+MTP，D 侧草稿层 conv state 逐字节对齐
- G3：NZ 排布，D 侧 cache 布局与 mooncake 一致
- G4：SWA 模型，D 侧只收窗口内 block
- G5：sparse 模型，rank 选择一致
- G6：SFA + DCP + prefix cache，K 块位置一致
- G7/G8/G9/G10：随各自主项

> 注：本设计基于 mooncake 代码静态分析与 HIXL block 寻址语义推导，未在 NPU 上实测。行号为 `vllm-ascend-v0.23.0` 当前状态，实施前需复核。MLA/compress（G1）是最大工程，建议单独出详细实施计划（类比 phase3-plan）。
