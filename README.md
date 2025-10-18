# Development and Evaluation of a Novel Framework Integrating LLMs, GNNs, and RL for Network Traffic Prediction and Optimization

## Overview
This project provides an end-to-end research framework integrating:
1. LLM-based semantic parsing of network traffic logs.
2. Graph Neural Network (PyTorch Geometric) for traffic load prediction.
3. Reinforcement Learning (Stable-Baselines3) for routing / resource optimization.
4. Integrated loop for adaptive decision making.

## Key Features
- Modular architecture (`src/llm`, `src/gnn`, `src/rl`, `src/pipeline`).
- Synthetic data generation and hooks for real datasets (CAIDA, NSL-KDD).
- Config-driven experimentation via `configs/config.yaml`.
- Reproducible training scripts.
- Evaluation metrics and visualization utilities.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Synthetic Data Generation
```bash
python data/synthetic/generate_synthetic.py --num-nodes 12 --timesteps 500
```
Outputs:
- `data/processed/traffic_timeseries.parquet`
- `data/processed/topology_edges.csv`
- `data/processed/logs.jsonl`

## Train GNN
```bash
python src/gnn/train_gnn.py --config configs/config.yaml
```

## Train RL Agent
```bash
python src/rl/train_rl.py --config configs/config.yaml
```

## Full Pipeline Loop
```bash
python src/pipeline/integrated_loop.py --config configs/config.yaml
```

## Evaluation
Artifacts:
- Metrics JSON in `outputs/reports/`
- Plots in `outputs/figures/`

## Real Datasets
You may integrate:
- CAIDA Anonymized Internet Traces (requires agreement)
- NSL-KDD (intrusion detection semantics)
- MAWI traffic archives
- RIPE Atlas ping/latency for augmentation

Convert raw data into:
- `traffic_timeseries.parquet` (columns: time, src, dst, bytes, packets, protocol, ...)
- `topology_edges.csv` (src, dst, capacity)
- `logs.jsonl` (raw textual lines for LLM parsing)

## Research Extensions
- Replace static GNN with temporal architectures (TGN, DCRNN).
- Multi-objective RL (Pareto frontier of latency vs energy).
- Graph-level policy networks with graph attention policy encoding.
- Continual learning for concept drift.

## Reproducibility
- Seed control in config.
- Deterministic PyTorch flags (note: full determinism may degrade GPU performance).
- Logged environment + model hyperparameters.

## Citation (Template)
```
@misc{yourproject2025,
  title={Development and Evaluation of a Novel Framework Integrating LLMs, GNNs, and RL for Network Traffic Prediction and Optimization},
  author={Your Name},
  year={2025},
  url={https://example.com}
}
```

## License
Apache 2.0 (adjust as needed).