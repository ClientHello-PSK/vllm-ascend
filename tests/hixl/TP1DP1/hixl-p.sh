#!/bin/bash
# ============================================================================
# HIXLConnectorV1 Phase 1 测试 - Prefill 节点（kv_producer）
# ----------------------------------------------------------------------------
# 最小配置：单 P 单 D、TP=1/PP=1/DP=1、单 HIXLConnectorV1（不组合）、
# 标准 FullAttention 模型。用于验证 HIXL 主链路（register_blocks_cache +
# link_clusters + pull_blocks）。
#
# 启动顺序：先启 hixl-p.sh（本脚本），等 P 端 log 出现
#   "HIXL KVCacheSendingThread listening on ..." 后，再启 hixl-d.sh。
#
# 验证点（P 端 log）：
#   1. "HixlDataDist initialized: role=PROMPT cluster_id=1000 ..."
#   2. "HIXL KVCacheSendingThread listening on tcp://<ip>:<handshake_port>"
#   3. 收到 D 的 GET_META / DONE 后正常回 ACK
# ============================================================================

# ======================== 需修改：环境相关 ========================
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$ASCEND_CUSTOM_OPP_PATH

MASTER_IP_ADDRESS="x.x.x.223"   # TODO: 改成 P 节点 IP
IP_ADDRESS="x.x.x.223"          # TODO: 改成 P 节点 IP（本机）
SERVICE_PORT=8800

# 卡：TP=1 用单卡
NETWORK_INTERFACE="eth0"
export ASCEND_RT_VISIBLE_DEVICES=7

# 模型：标准 FullAttention 单 group（HIXL Phase 1 要求 block_size_scale==1、单 group）
# TODO: 改成你的标准 attention 模型路径（Qwen2.5/Llama 等，勿用 MoE/Mamba/MTP）
MODEL_PATH="/data/models/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B"

# Phase 1 并行维度（硬约束：TP=1/PP=1，建议 DP=1）
PREFILL_DATA_PARALLEL_SIZE=1
PREFILL_TENSOR_PARALLEL_SIZE=1
DECODE_DATA_PARALLEL_SIZE=1
DECODE_TENSOR_PARALLEL_SIZE=1

LIBJEMALLOC_SO_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
GPU_MEMORY_USE=0.9
MAX_BATCHED_TOKENS=4096

# ======================== 通用环境变量（无 mooncake）========================
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
  --kv-cache-memory-bytes 52162245120 \
  --kv-transfer-config \
'{
  "kv_connector": "HIXLConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "21299",
  "kv_connector_extra_config": {
    "hixl": {
      "cluster_id_base": 1000,
      "model_id": 0,
      "link_timeout_ms": 5000
    },
    "prefill": {"dp_size": 1, "tp_size": 1},
    "decode":   {"dp_size": 1, "tp_size": 1}
  }
}' \
  > /tmp/hixl-p.log 2>&1 &

echo "Prefill (kv_producer) started, log: /tmp/hixl-p.log, pid: $!"
