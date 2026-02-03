!#/bin/bash

apt-get update
apt-get install libnuma1

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv venv -p 3.12
uv pip install -r requirements.txt
