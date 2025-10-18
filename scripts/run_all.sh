#!/usr/bin/env bash
set -e

python data/synthetic/generate_synthetic.py --num-nodes 12 --timesteps 400
python src/llm/traffic_parser.py --logs data/processed/logs.jsonl --out data/processed/log_embeddings.pt
python src/gnn/train_gnn.py --config configs/config.yaml
python src/rl/train_rl.py --config configs/config.yaml
python src/pipeline/integrated_loop.py --config configs/config.yaml