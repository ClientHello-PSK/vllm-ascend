#!/bin/bash 

export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:$ASCEND_CUSTOM_OPP_PATH

# ========================  
# 服务可配置变量
# ========================
export VLLM_VERSION=0.21.0
MASTER_IP_ADDRESS="x.x.x.223" # 主节点的IP地址
IP_ADDRESS="x.x.x.223"        # 当前节点的IP地址
SERVICE_PORT=8600

#export HCCL_DETERMINISTIC=true

#mooncake相关
export MOONCAKE_CONFIG_PATH=./mooncake.json
# export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/python/site-packages/mooncake:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake:$LD_LIBRARY_PATH 
# 分层调度
# export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=1

# 大融合算子
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

# 卡相关
NETWORK_INTERFACE="eth0"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 主节点为0，从节点依次递增
DECODE_DATA_PARALLEL_RANK=0

PREFILL_DATA_PARALLEL_SIZE=2
PREFILL_TENSOR_PARALLEL_SIZE=8

DECODE_DATA_PARALLEL_SIZE=2
DECODE_TENSOR_PARALLEL_SIZE=8


# ========================  
# 模型配置变量 
# ========================
MODEL_PATH="/data/models/MiniMax-M2.7-26.1.RC2"  # 模型权重路径 
MODEL_NAME="MiniMax-M2.7"  # 服务模型名称
TOOL_CALL_PARSER="minimax_m2" # 工具调用
REASONING_PARSER="minimax_m2" # 思考解析

LIBJEMALLOC_SO_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
GPU_MEMORY_USE=0.95

# chunked-size大小
MAX_BATCHED_TOKENS=4096

# 开启MC2算子需要配置适当的HCCL_BUFFSIZE大小
export HCCL_BUFFSIZE=512


# ========================  
# 功能参数（公共）  
# ========================  
export VLLM_USE_V1=1  


# ========================  
# 性能优化参数（公共）  
# ========================
export OMP_NUM_THREADS=100  
export OMP_PROC_BIND=false  
export TASK_QUEUE_ENABLE=1    # 是否开启算子下发优化：0关闭；1或者未配置走Level 1优化；2走Level 2优化
export LD_PRELOAD=$LIBJEMALLOC_SO_PATH:$LD_PRELOAD  
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  


# ========================  
# 配置参数（公共）  
# ========================  
export HCCL_IF_IP="$IP_ADDRESS"  
export GLOO_SOCKET_IFNAME="$NETWORK_INTERFACE"  
export TP_SOCKET_IFNAME="$NETWORK_INTERFACE"  
export HCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"  
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# "enable_weight_nz_layout"： 是否将量化权重转换为NZ格式以加速矩阵乘法

# ========================  
# 正常DP|DP外置  
# ========================
# 正常DP场景
# 配置data-parallel-size、data-parallel-size-local、data-parallel-start-rank

# --data-parallel-hybrid-lb, --no-data-parallel-hybrid-lb 负载均衡，与--data-parallel-start-rank 明确结合设置，默认是False

# DP外置场景
# data-parallel-size (DP数值)
# data-parallel-rank (DP的rank，从0开始) [配置后启用DP外置]


# ========================  
# 服务启动命令（归一化执行）  
# ========================  
vllm serve "$MODEL_PATH" \
--host "$IP_ADDRESS" \
--port "$SERVICE_PORT" \
--data-parallel-size "$DECODE_DATA_PARALLEL_SIZE" \
--data-parallel-rank "$DECODE_DATA_PARALLEL_RANK" \
--api-server-count 1 \
--data-parallel-address "$MASTER_IP_ADDRESS" \
--data-parallel-rpc-port 8082 \
--tensor-parallel-size "$DECODE_TENSOR_PARALLEL_SIZE" \
--served-model-name "$MODEL_NAME" \
--max_model_len 196608 \
--max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
--max-num-seqs 16 \
--trust-remote-code \
--gpu_memory_utilization "$GPU_MEMORY_USE" \
--enable-auto-tool-choice \
--tool-call-parser "$TOOL_CALL_PARSER" \
--reasoning-parser "$REASONING_PARSER" \
--enable-expert-parallel \
--no-enable-prefix-caching \
--enforce-eager \
--kv-transfer-config \
'{
	"kv_connector": "MultiConnector",
	"kv_role": "kv_producer",
	"kv_connector_extra_config": {
		"connectors": [{
				"kv_connector": "MooncakeConnectorV1",
				"kv_role": "kv_producer",
				"kv_port": "21202",
				"kv_connector_extra_config": {
					"prefill": {
						"dp_size": '"$PREFILL_DATA_PARALLEL_SIZE"',
						"tp_size": '"$PREFILL_TENSOR_PARALLEL_SIZE"'
					},
					"decode": {
						"dp_size": '"$DECODE_DATA_PARALLEL_SIZE"',
						"tp_size": '"$DECODE_TENSOR_PARALLEL_SIZE"'
					}
				}
			}, {
				"kv_connector": "AscendStoreConnector",
				"kv_role": "kv_producer",
				"kv_connector_extra_config": {
					"lookup_rpc_port": "0",
					"backend": "mooncake"
				}
			}
		]
	}
}' \
--speculative-config '{"method":"minimax_m2_mtp","model":"/data/lxxxx/saved_mtp_dense/m27_mtp3_0519","num_speculative_tokens":3}' \
> /data/dxxxx/test/202606-m2.7/log/minimax.log 2>&1 &

#--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/data/sxxxx/mtp_layerwise_v0.18/profile", "torch_profiler_with_stack": false, "torch_profiler_record_shapes":true}' \
#--speculative-config '{"method":"minimax_m2_mtp","model":"/opt/models/GTSLLM-Pro-Moe-Code-910B-26.5.0/GTSLLM-Pro-Moe-Code-910B-26.5.0/mtp_dense","num_speculative_tokens":1}' \
