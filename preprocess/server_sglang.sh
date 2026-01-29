export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR='/data/oceanus_share/yangqianwen/test_sglang/expert_deepseek'

set -x
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
model_path='./llm_model/DeepSeek-R1-0528'
served_model_name=default 
#`basename ${model_path}`
echo ${served_model_name}


echo ${model_path}

python -m sglang.launch_server \
        --model-path ${model_path} \
        --mem-fraction-static 0.85 \
        --max-prefill-tokens 50000 \
        --chunked-prefill-size 2048 \
        --dist-init-addr "$MLP_WORKER_0_HOST:5000" \
        --nnodes 2 \
        --node-rank $MLP_ROLE_INDEX \
        --tp 16 \
        --dp 1 \
        --trust-remote-code \
        --log-level debug \
        --host 0.0.0.0 \
        --port 8000 \
        --watchdog-timeout 2000 \
        --served-model-name ${served_model_name} \
        --expert-distribution-recorder-mode per_token \
        --expert-distribution-recorder-buffer-size 1000000 \
        --disable-cuda-graph 


       