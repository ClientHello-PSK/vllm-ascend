# RFC: vLLM-ascend HIXL Connector — KV Cache P2P 传输直连需求


## 1. 需求描述

**背景**：vLLM-ascend 在 P/D（Prefill/Decode）分离场景下，P 节点算好的 KV Cache 需搬到 D 节点以避免重算，由 P2P connector 在 scheduler↔worker 之间协调完成搬运。现有三种 connector（MooncakeConnectorV1 / Hybrid / Layerwise）均基于 mooncake TransferEngine，其中 `MooncakeConnectorV1` 能力最全、是主用实现。

mooncake 的 ascend 后端底层已是 HIXL（CANN 单边通信库），在其上叠加了 store/master/lease 等中间层；对 P/D 直传而言这些中间层不带来价值。更关键的是，mooncake 数据面采用**字节寻址**——按字节段注册内存、按字节地址读写，而 vLLM 的 KV cache 是 **block 粒度**的 paged tensor。两者模型不一致，需由 connector 在上层维护 block 与字节地址的换算：维护 block 到字节地址的映射、逐 block 推算字节地址、TP>1 时用字节偏移拼接多 rank shard 后重排。

**要实现的特性**：新建一个 `HIXLConnector`，绕过 mooncake 字节寻址数据面，直接对接 HIXL 的 LLM-DataDist 能力承担 P/D 间 KV 传输，与 mooncake 并存、可配置切换、渐进迁移。

**期望解决的问题**：

- 消除 block 与字节地址的换算逻辑（地址换算、region 合并、TP 偏移拼接）；
- 去掉 mooncake store/master/lease 中间层开销；
- 寻址、block 划分与传输语义统一在 block 层面，降低耦合。

> 现状拆解见 §6 技术方案前置的 mooncake 现状分析；本需求对标基准为 `MooncakeConnectorV1`。


## 2. 功能要点

- [ ] **block 索引寻址替代字节寻址**：D 按 block id 主动拉取，不维护字节地址与 block 的换算映射。
- [ ] **保留 mooncake 控制面**：ZMQ 握手、scheduler 决策、block 范围计算、端口分配、延迟释放全部沿用；数据面载荷由"字节地址 + 会话标识"替换为"集群标识 + 监听端口"。
- [ ] **显式建链替代隐式会话路由**：D 单边向 P 建立链路，P 全程被动，以显式链路状态替代字符串会话标识的隐式路由。
- [ ] **保留 D-pull 模型**：P 仅暴露可远程访问的 block 缓存供 D 主动拉取，传输发起方与数据流向不变。
- [ ] **TP>1 staging + 后置重排**：block 寻址接口只能整块写入，无法直写 block 内部 split，须改用暂存缓存加后置转置重排。
- [ ] **寻址相关能力重新设计**：覆盖 MLA/compress 的 kernel 展开、Mamba state 的 conv_padding、NZ layout 的 reformat、`r_blk>1` 的 kernel_size 整除。
- [ ] **寻址无关能力复用**：SWA 滑窗裁剪、sparse attention、SFA DCP replicate-K、CP+MLA 联合 kernel 的映射逻辑。
- [ ] **与 mooncake 并存**：作为独立 connector 与 mooncake 并存，可配置切换，不破坏现有 mooncake 用户。
- [ ] **与 mooncake 同 P/D 并行配置下逐位对齐 KV**（验收基线）。


## 3. 上下游影响与接口诉求

本需求上下游均为既有系统：上游 vLLM / vLLM-ascend 调度与 KV cache 框架、HIXL（即 LLM-DataDist）、下游 HCCL / CANN 单边通信。本需求消费其已有能力，不要求新增接口，下文不重复展开既有 API 规格，仅列需上下游配合或受其约束的诉求点。

### 3.1 上游：vLLM / vLLM-ascend

connector 作为既有插件接入，遵循 `KVConnectorBase_V1` 调度逻辑，不修改上游接口，与 mooncake 并存。须线程安全（调度决策在调度线程，数据面在独立收发线程，以 `remote_metadata_lock` 等锁保护）；沿用 vLLM logger 关键路径打点（建链、拉取、释放）。block 粒度由 `KVCacheConfig` 决定，多维并行（TP / PP / PCP / DCP）规格随上游配置。

### 3.2 HIXL（LLM-DataDist）

本需求消费 LLM-DataDist 既有 API（`init` / `register_blocks_cache` / `pull_blocks` / `link_clusters` 等），不要求新增接口。其既有行为构成以下约束，须在 connector 侧规避：

- **整块写入**：`pull_blocks` 只能整 block 写入，不支持 block 内 split，TP>1 须 staging + 后置重排。
- **两维缓存键**：`BlocksCacheKey` 仅 (cluster_id, model_id)，无 group 维度，重复键 last-wins 覆盖；须由 connector 保证每组（含 mamba conv / ssm 子 cache）唯一。
- **单 shape 约束**：`CacheDesc` 单 shape，异质 state（conv 2D vs ssm 3D）无法共载，须拆独立 cache。
- **幂等建链**：`link_clusters` 对已建链 cluster 须 no-op、线程安全，失败抛明确异常。
- **remote_accessible**：P / D 均须置 True，否则 `pull_blocks` 返回 `LLM_PARAM_INVALID`。
- **状态码分类**：对 `LLM_PARAM_INVALID` / `LLM_FAILED` 做分类与可重试判定。

性能诉求：拉取延迟与吞吐不劣于 mooncake 基线，待 NPU 实测对照。资源约束：block 寻址下 HCCL region 上限是否较 mooncake 缓解待实测。可观测性：建链状态、缓存键分配、拉取计数需可观测，便于排查 last-wins 覆盖等静默错误。

### 3.3 下游：HCCL / CANN

LLM-DataDist 底层经 HCCL 单边 RDMA 读写，由 native 库封装，connector 不直接调用；依赖 NPU 驱动与 CANN 版本，受每进程 RDMA 资源（region 数、链路数）配额约束。


## 4. 对外 API 变更

### 4.1 vLLM-ascend 侧（新增）

- **新增 connector 类**：`HIXLConnector`，注册为 `kv_connector` 取值；与 mooncake 并存，默认行为不变（仅当显式配置 `kv_connector=HIXLConnector` 时启用）。
- **新增配置项**（`kv_connector_extra_config`）：
  - `hixl` 段：`cluster_id_base`（必填）、`listen_port_base`、`model_id`、`link_timeout_ms`、`llm_options`（透传 `ge_options`）、`link_total_time` / `link_retry_count` / `sync_kv_timeout`（可选）。
  - `prefill` / `decode` 段：`tp_size` / `dp_size` / `pp_size` / `pp_layer_partition`。
- **新增结构体**：`HixlAgentMetadata`，承载 P→D 暴露的 cluster_id / listen_ip / listen_port / group2layer 等，替代 mooncake 的字节地址字段集。
- **固定写入 LLM-DataDist `LLMConfig` 的常量**（非用户配置项）：`transfer_backend="hixl"`、`local_comm_res=""`。
- **兼容性**：对 mooncake 用户零影响（独立 connector，不修改 mooncake 代码路径）；需同步更新对外配置文档与 connector 列表。

### 4.2 HIXL / LLM-DataDist 侧

- **不涉及新增对外 API**：本需求消费 LLM-DataDist 既有 API（`init` / `register_blocks_cache` / `pull_blocks` / `link_clusters` / `BlocksCacheKey` / `CacheDesc`）。
- **能力诉求（非接口变更，见 §3.2）**：`pull_blocks` 整块写入限制、`BlocksCacheKey` 两维 last-wins 行为、`CacheDesc` 单 shape 约束——这些为既有行为，本需求在 connector 侧规避，不要求 HIXL 修改接口。
- **需同步更新文档**：若 HIXL 侧未来计划支持 sub-block 写入或多维缓存键，需在本 RFC §6 项 3/4 闭环后回填。


## 5. 对外 API 使用示例

> 新增vllm配置，不新增API，以下配置在 vLLM server 拉起时加载配置。

### 5.1 配置示例

P 侧（生产 KV）：

```jsonc
{
  "kv_role": "kv_producer",
  "kv_port": 8000,
  "kv_connector": "HIXLConnector",
  "kv_connector_extra_config": {
    "hixl": {
      "cluster_id_base": 1000,        // P 侧起始 cluster_id，须与 D 不相交
      "listen_port_base": 8000,
      "model_id": 0,
      "link_timeout_ms": 5000
    },
    "prefill": { "tp_size": 1, "dp_size": 1 }
  }
}
```

D 侧（消费 KV）：

```jsonc
{
  "kv_role": "kv_consumer",
  "kv_port": 8000,
  "kv_connector": "HIXLConnector",
  "kv_connector_extra_config": {
    "hixl": {
      "cluster_id_base": 2000,        // D 侧起始 cluster_id，与 P 不相交
      "listen_port_base": 8000,
      "model_id": 0,
      "link_timeout_ms": 5000
    },
    "decode": { "tp_size": 1, "dp_size": 1 }
  }
}
```

## 6. 技术方案

### 6.1 mooncake 现状（对标基准）

`MooncakeConnectorV1` 本质为基于字节寻址的 RDMA 传输中间件：底层仅以连续字节段注册与读写内存，不感知 vLLM 的 block 粒度、层结构与 TP 划分。connector 需在二者间维护 block 与字节地址的换算，即本方案拟消除的全部成本。按职责分三层：

- **控制面**（与传输引擎无关，可复用）：ZMQ 控制面（P ROUTER / D REQ socket 池）、两类控制消息（元数据查询、拉取完成通知）、scheduler 决策、block 偏移计算、端口分配、延迟释放与超时强制释放。
- **数据面**（与传输引擎强耦合，待删除）：内存注册（字节段 + region 合并规避 256 上限）、地址计算（base_addr/len/stride 推算字节地址）、数据拼接（TP>1 字节偏移拼 split + 后置转置）、对端路由（字符串会话标识）。
- **覆盖能力清单**（按寻址方式二分）：寻址无关（SWA / sparse / SFA / CP 映射，复用）；寻址相关（MLA 展开 / Mamba conv_padding / NZ reformat / r_blk 整除，须重设计）。

### 6.2 HIXL Connector 数据面

- **P 侧暴露**：`register_blocks_cache` 按 block 注册 KV cache，`remote_accessible=True`，被动等待；启动 ZMQ ROUTER 监听握手端口。
- **D 侧拉取**：通过 ZMQ 取 P 的 `HixlAgentMetadata`（含 cluster_id / listen_ip / listen_port / group2layer），`ensure_linked` 建链后 `pull_blocks` 按 block id 主动拉取。
- **TP>1 staging + 后置重排**：block 寻址接口只能整块写入，无法直写 split，改用暂存缓存接收各 P rank shard，各 shard 全部写入后，后置转置重排至 D 真实 cache。

### 6.3 时序图

![alt text](流程图.png)

## 7. 测试方案

- **TP=1 逐位对齐**：同 P/D 配置下，HIXL 与 mooncake 拉取结果逐 block 比对。
- **TP>1 staging**：多 P rank shard 写入 staging 缓存后转置重排，结果与 mooncake 字节偏移直写一致。
- **MLA/compress**：logical→kernel 展开与拉取 block id 口径一致；compress + D 端 prefix cache 起点错位复现与修复验证。
- **Mamba state**：conv/ssm 作为独立 cache 注册、缓存键不冲突；TP>1 断言生效。
- **NZ layout**：pull 写入 D cache 后 NZ reformat 等价；MLA 单 head + TP=1 直接写入 D cache。
- **SWA / sparse / SFA / CP**：寻址无关能力与 mooncake 同配置逐位对齐。
- **r_blk>1**：P/D block_size 不等场景；scale==1 下不可达分支验证。
- **region 配额**：block 寻址下 256 上限约束是否缓解（对比 mooncake）。
- **并存与切换**：`kv_connector` 在 mooncake / HIXL 间切换，互不破坏。
- **异常**：建链超时、`pull_blocks` 返回 `LLM_PARAM_INVALID`/`LLM_FAILED`、缓存键 last-wins 覆盖的可观测性与处理。


## 8. 验收标准

- [ ] 同 P/D 并行配置下，HIXL 与 mooncake 拉取的 KV 逐位对齐。
- [ ] block 与字节地址的换算逻辑（base_addr/len/stride 维护、region 合并、TP 字节偏移拼接）全部移除。
- [ ] 控制面（ZMQ 握手、scheduler 决策、block 偏移、端口分配、延迟释放）原样复用。
- [ ] TP>1 经 staging + 后置重排正确还原，不劣于 mooncake 字节偏移方案。
- [ ] MLA/compress、Mamba state、NZ layout、r_blk>1 在 block 寻址下重新设计并通过同配置对齐。
- [ ] SWA/sparse/SFA/CP 映射寻址无关能力原样复用。
- [ ] `HIXLConnector` 与 mooncake 并存，配置切换不破坏现有用户。
- [ ] 配置项（`hixl`/`prefill`/`decode` 段）文档与代码一致，`kv_role`/`kv_port`/`cluster_id_base` 等字段语义明确。
- [ ] 异常路径（建链失败、拉取失败、缓存键冲突）有明确状态码与可观测日志。
- [ ] §6.4 关键约束与风险逐项给出结论（实测/固化/上游修复）。
