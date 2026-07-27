#!/bin/bash
# ============================================================================
# HIXLConnectorV1 Phase 2 测试 - Decode 节点（kv_consumer），TP=2
# ----------------------------------------------------------------------------
# 相对 TP1DP1 的改动：
#   - ASCEND_RT_VISIBLE_DEVICES 由单卡改为 2 卡
#   - PREFILL/DECODE_TENSOR_PARALLEL_SIZE 1 -> 2
#   - kv_connector_extra_config.prefill/decode.tp_size 1 -> 2
# Phase 2 已放开 hixl_connector.py:1016 的 tp_size==1 断言；staging +
# reformat 链路（plan §4.4/§4.5）在 num_group_pulls=2 下触发。
#
# 启动顺序：确认 hixl-p.sh 已启动且 P 端 ZMQ ROUTER 已监听后再启本脚本。
#
# 验证点（D 端 log）：
#   1. "HixlDataDist initialized: role=DECODER cluster_id=2000 ..."
#   2. "HIXL linked remote_cluster_id=1000 (<P_IP>:<P_listen_port>)"
#      —— 每个 P tp_rank 各 link 一次（Phase 2 plan §2.5/§4.6）
#   3. "HIXL pull ok. request=... group=... remote_cluster=1000 ..."
#      —— 出现 num_group_pulls=2 的多 rank 拉取 + reformat 拼 head
#   4. 输出 token 与同配置（TP=2）MooncakeConnectorV1 一致（正确性验收）
# ============================================================================

# ======================== 需修改：环境相关 ========================
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$ASCEND_CUSTOM_OPP_PATH

MASTER_IP_ADDRESS="x.x.x.225"   # TODO: 改成 D 节点 IP
IP_ADDRESS="x.x.x.225"          # TODO: 改成 D 节点 IP（本机）
SERVICE_PORT=8801

# [Phase2] TP=2 用 2 张卡
NETWORK_INTERFACE="eth0"
export ASCEND_RT_VISIBLE_DEVICES=6,7

# 模型：须与 P 端一致（同路径/同配置）
MODEL_PATH="/data/models/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B"

# [Phase2] Phase 2 并行维度：TP 1 -> 2，PP 仍 =1（Phase 3 才放开）
PREFILL_DATA_PARALLEL_SIZE=1
PREFILL_TENSOR_PARALLEL_SIZE=2
DECODE_DATA_PARALLEL_SIZE=1
DECODE_TENSOR_PARALLEL_SIZE=2

LIBJEMALLOC_SO_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
GPU_MEMORY_USE=0.9
MAX_BATCHED_TOKENS=64   # decode 用小 batched tokens

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
# 注意：去掉 --kv-cache-memory-bytes（TP1DP1 写死的总量在 TP=2 下会 OOM），
#       改由 --gpu_memory_utilization 自动划分每卡 KV。
vllm serve "$MODEL_PATH" \
  --host "$IP_ADDRESS" \
  --port "$SERVICE_PORT" \
  --data-parallel-size "$DECODE_DATA_PARALLEL_SIZE" \
  --tensor-parallel-size "$DECODE_TENSOR_PARALLEL_SIZE" \
  --served-model-name "$MODEL_NAME" \
  --max_model_len 32768 \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs 16 \
  --trust-remote-code \
  --gpu_memory_utilization "$GPU_MEMORY_USE" \
  --enforce-eager \
  --kv-transfer-config \
'{
  "kv_connector": "HIXLConnectorV1",
  "kv_role": "kv_consumer",
  "kv_port": "21299",
  "kv_connector_extra_config": {
    "hixl": {
      "cluster_id_base": 2000,
      "model_id": 0,
      "link_timeout_ms": 5000
    },
    "prefill": {"dp_size": 1, "tp_size": 2},
    "decode":   {"dp_size": 1, "tp_size": 2}
  }
}' \
  > /tmp/hixl-d.log 2>&1 &

echo "Decode (kv_consumer) started, log: /tmp/hixl-d.log, pid: $!"
