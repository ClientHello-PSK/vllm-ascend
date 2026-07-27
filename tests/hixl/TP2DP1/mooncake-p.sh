#!/bin/bash
# ============================================================================
# MooncakeConnectorV1 基线测试 - Prefill 节点（kv_producer），TP=2
# ----------------------------------------------------------------------------
# 作为 HIXL TP2DP1 的对比基线：与 hixl-p.sh 严格对齐（同 Qwen2.5-3B / TP=2 / DP=1）。
# 验证 HIXL 输出与 mooncake 一致（HIXL Phase 2 正确性验收的基准）。
#
# 相对 TP1DP1/mooncake-p.sh 的改动：
#   - ASCEND_RT_VISIBLE_DEVICES 单卡 -> 2 卡
#   - PREFILL/DECODE_TENSOR_PARALLEL_SIZE 1 -> 2
#   - kv_connector_extra_config.prefill/decode.tp_size 1 -> 2
#
# ⚠️ mooncake 必需（HIXL 不需要这些，别漏）：MOONCAKE_CONFIG_PATH + mooncake 库 +
#    mooncake.json，缺一启动就崩。
#
# 启动顺序：先启本脚本，P 端 log 出现 "KVCacheSendingThread listening on ..." 后，
#           再启 mooncake-d.sh。
# ============================================================================

# ======================== 需修改：环境相关 ========================
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$ASCEND_CUSTOM_OPP_PATH

MASTER_IP_ADDRESS="x.x.x.223"   # TODO: 改成 P 节点 IP
IP_ADDRESS="x.x.x.223"          # TODO: 改成 P 节点 IP（本机）
SERVICE_PORT=8800

# ---- mooncake 必需（HIXL 不需要这些，别漏）----
export MOONCAKE_CONFIG_PATH=./mooncake-p.json
# TODO: 改成你环境的 mooncake 库路径（参考原 p0-223.sh，cann 版本号按实际）
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake:$LD_LIBRARY_PATH

# [Phase2] TP=2 用 2 张卡；启动前 npu-smi info 确认该卡空闲
NETWORK_INTERFACE="eth0"
export ASCEND_RT_VISIBLE_DEVICES=6,7

# 模型：与 hixl-p.sh 一致（标准 FullAttention，便于对比）
MODEL_PATH="/data/models/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B"

# [Phase2] 与 hixl-p.sh 一致（TP 1 -> 2）
PREFILL_DATA_PARALLEL_SIZE=1
PREFILL_TENSOR_PARALLEL_SIZE=2
DECODE_DATA_PARALLEL_SIZE=1
DECODE_TENSOR_PARALLEL_SIZE=2

LIBJEMALLOC_SO_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
GPU_MEMORY_USE=0.9
MAX_BATCHED_TOKENS=4096

# ======================== 通用环境变量 ========================
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=512
export HCCL_IF_IP="$IP_ADDRESS"
export GLOO_SOCKET_IFNAME="$NETWORK_INTERFACE"
export TP_SOCKET_IFNAME="$NETWORK_INTERFACE"
export HCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=100
export OMP_PROC_BIND=false
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=$LIBJEMALLOC_SO_PATH:$LD_PRELOAD

# ======================== 启动命令 ========================
vllm serve "$MODEL_PATH" \
  --host "$IP_ADDRESS" \
  --port "$SERVICE_PORT" \
  --data-parallel-size "$PREFILL_DATA_PARALLEL_SIZE" \
  --tensor-parallel-size "$PREFILL_TENSOR_PARALLEL_SIZE" \
  --served-model-name "$MODEL_NAME" \
  --max_model_len 32768 \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs 16 \
  --trust-remote-code \
  --gpu_memory_utilization "$GPU_MEMORY_USE" \
  --enforce-eager \
  --kv-transfer-config \
'{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "21202",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": 1, "tp_size": 2},
    "decode":   {"dp_size": 1, "tp_size": 2}
  }
}' \
  > /tmp/mooncake-p.log 2>&1 &

echo "Prefill (MooncakeConnectorV1 kv_producer) started, log: /tmp/mooncake-p.log, pid: $!"
