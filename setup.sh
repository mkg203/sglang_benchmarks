git checkout long_context_exp

apt-get update
apt-get install libnuma1

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync

if [[ -n $HF_TOKEN ]]; then
  hf auth login --token $HF_TOKEN
else
  echo "hugging face cli not initialized"
fi
