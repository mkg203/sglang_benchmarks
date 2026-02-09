python -m sglang.launch_server \
        --model-path meta-llama/Llama-3.2-1B \
        --port 30000 \
        --mem-fraction-static 0.4 \
        --cpu-offload-gb 4 \
      --enable-metrics
