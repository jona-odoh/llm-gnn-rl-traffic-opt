import argparse
import yaml
import torch
import networkx as nx
import numpy as np
from pathlib import Path
from src.llm.traffic_parser import LogLLMEmbedder, load_logs
from src.gnn.model import EdgeRegressionGNN
from src.rl.env import RoutingEnv
from stable_baselines3 import PPO
import json


def build_graph_from_topology(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row.src, row.dst, capacity=row.get("capacity", 100.))
    return G


def simulate_new_logs(num_lines: int = 20):
    protocols = ["TCP", "UDP", "ICMP", "ALERT", "DENIED"]
    logs = []
    rng = np.random.default_rng(123)
    for _ in range(num_lines):
        p = rng.choice(protocols, p=[0.4, 0.3, 0.1, 0.1, 0.1])
        logs.append(
            f"{p} src=10.0.0.{rng.integers(1, 50)} dst=10.0.1.{rng.integers(1, 50)} bytes={rng.integers(50, 5000)}")
    return logs


def load_rl_model(models_dir: Path, env):
    """Load RL model with dimension verification"""
    # Look for any PPO model
    model_files = list(models_dir.glob("rl_PPO*.zip"))
    if not model_files:
        raise FileNotFoundError(
            "No RL model found. Please run train_rl.py first.")

    model_path = model_files[0]
    print(f"🔄 Loading RL model: {model_path.name}")

    # Load environment specs if available
    env_specs_path = models_dir / "env_specs.json"
    if env_specs_path.exists():
        with open(env_specs_path, "r") as f:
            env_specs = json.load(f)
        print(f"📋 Model was trained with: {env_specs}")

    model = PPO.load(model_path)

    # Verify dimensions match
    env_obs, _ = env.reset()
    env_obs_dim = env_obs.shape[0]
    model_obs_dim = model.observation_space.shape[0]

    print(f"🔍 Environment observation dim: {env_obs_dim}")
    print(f"📦 Model observation space dim: {model_obs_dim}")

    if env_obs_dim != model_obs_dim:
        raise ValueError(
            f"Observation dimension mismatch!\n"
            f"  Environment: {env_obs_dim}\n"
            f"  Model: {model_obs_dim}\n"
            f"Solution: Retrain the model with current environment setup."
        )

    print(f"✅ Model loaded successfully")
    return model


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])

    for p in [reports_dir, figures_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Load GNN
    gnn_model = EdgeRegressionGNN(
        node_in=1,
        edge_in=3,
        hidden=cfg["gnn"]["hidden_dim"],
        out_dim=cfg["gnn"]["output_dim"],
        num_layers=cfg["gnn"]["num_layers"],
        dropout=cfg["gnn"]["dropout"]
    ).to(device)
    gnn_model.load_state_dict(torch.load(
        models_dir / "best_gnn.pt", map_location=device))
    gnn_model.eval()

    # Build graph - USE SAME GRAPH AS TRAINING
    from src.rl.train_rl import build_graph
    G = build_graph(num_nodes=12, p=0.2, seed=cfg["seed"])

    # Graph adjustment to match training exactly
    desired_num_edges = 28
    edges = list(G.edges())
    if len(edges) > desired_num_edges:
        edges_to_remove = edges[desired_num_edges:]
        G.remove_edges_from(edges_to_remove)

    # Initialize environment with SAME parameters as training
    env = RoutingEnv(
        G,
        num_flows=5,
        candidates_per_flow=3,
        max_queue=cfg["environment"]["max_queue"],
        latency_base=cfg["environment"]["latency_base"],
        congestion_threshold=cfg["environment"]["congestion_threshold"],
        penalty_congestion=cfg["environment"]["penalty_congestion"],
        reward_weights=cfg["environment"]["reward_weights"],
        seed=cfg["seed"]
    )

    # Load RL model with verification
    rl_model = load_rl_model(models_dir, env)

    # Reset environment to get initial observation
    obs, _ = env.reset()

    # LLM embedder (for simulated new logs)
    embedder = LogLLMEmbedder(
        model_name=cfg["llm"]["model_name"], max_length=cfg["llm"]["max_length"])

    loop_metrics = []
    for step in range(50):
        # Simulate incoming logs (in practice: read streaming logs)
        new_logs = simulate_new_logs()
        embeddings = embedder.embed_texts(
            new_logs, batch_size=cfg["llm"]["batch_size"])
        semantic_signal = embeddings.mean(dim=0).numpy()  # Example aggregation

        # Optionally integrate semantic signal into environment or GNN features
        # For demonstration, adjust environment edge utilization baseline drift
        drift = np.tanh(semantic_signal.mean()) * 0.01
        env.edge_utilization = np.clip(
            env.edge_utilization * (1 + drift), 0, 2.0)

        # RL selects action
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        info["step"] = step
        info["semantic_drift"] = float(drift)
        loop_metrics.append(info)

        print(
            f"Step {step}: Reward={reward:.2f}, Latency={info['latency']:.2f}, Throughput={info['throughput']:.2f}")

        if term or trunc:
            obs, _ = env.reset()

    with open(reports_dir / "pipeline_loop_metrics.json", "w") as f:
        json.dump(loop_metrics, f, indent=2)
    print("✅ Integration loop complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
