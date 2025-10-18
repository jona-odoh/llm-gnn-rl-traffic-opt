#!/usr/bin/env bash
set -e
python data/synthetic/generate_synthetic.py --num-nodes 10 --timesteps 300
python src/llm/traffic_parser.py --logs data/processed/logs.jsonl --out data/processed/log_embeddings.pt