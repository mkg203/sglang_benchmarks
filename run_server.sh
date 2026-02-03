!#/bin/bash

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B \
  --port 30000 \
  --enable-metrics \
  # --enable-hierarchical-cache

