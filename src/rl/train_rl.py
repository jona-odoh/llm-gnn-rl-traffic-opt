import argparse
import yaml
import torch
import networkx as nx
from src.rl.env import RoutingEnv  # FIXED IMPORT
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from src.rl.visualize import visualize_routing, animate_routing_over_time, create_pdf_report
from pathlib import Path
import json
import numpy as np
import time


class MetricsCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, save_path: Path, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.metrics = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            total_r = 0
            infos = []
            for _ in range(20):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.eval_env.step(action)
                total_r += r
                infos.append(info)
                if term or trunc:
                    obs, _ = self.eval_env.reset()
            avg_latency = np.mean([i["latency"] for i in infos])
            avg_throughput = np.mean([i["throughput"] for i in infos])
            avg_loss = np.mean([i["packet_loss"] for i in infos])
            self.metrics.append({
                "steps": self.n_calls,
                "avg_reward": total_r / len(infos),
                "avg_latency": avg_latency,
                "avg_throughput": avg_throughput,
                "avg_packet_loss": avg_loss
            })
            with open(self.save_path / "rl_metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)
        return True


def build_graph(num_nodes=12, p=0.2, seed=42):
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and rng.random() < p:
                capacity = float(rng.integers(50, 200))
                G.add_edge(i, j, capacity=capacity)
    # Ensure connectivity fallback
    if not nx.is_weakly_connected(G):
        # connect chain
        for i in range(num_nodes - 1):
            if not G.has_edge(i, i + 1):
                G.add_edge(i, i + 1, capacity=100.0)
    return G


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    G = build_graph()

    # Graph adjustment here
    desired_num_edges = 28  # or whatever your target number is
    edges = list(G.edges())
    if len(edges) > desired_num_edges:
        edges_to_remove = edges[desired_num_edges:]
        G.remove_edges_from(edges_to_remove)
    # You can also prune nodes similarly if needed

    train_env = RoutingEnv(
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

    eval_env = RoutingEnv(
        G,
        num_flows=5,
        candidates_per_flow=3,
        seed=cfg["seed"] + 1
    )

    algo = cfg["rl"]["algo"].upper()
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if algo == "PPO":
        model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=cfg["rl"]["learning_rate"],
                    gamma=cfg["rl"]["gamma"])
    elif algo == "DQN":
        model = DQN("MlpPolicy", train_env, verbose=1, learning_rate=cfg["rl"]["learning_rate"],
                    gamma=cfg["rl"]["gamma"])
    else:
        raise ValueError(f"Unsupported algo {algo}")

    callback = MetricsCallback(eval_env, eval_freq=1000, save_path=reports_dir)
    model.learn(total_timesteps=cfg["rl"]
                ["total_timesteps"], callback=callback)

    # === ADDED: Environment Consistency Verification ===
    obs, _ = train_env.reset()
    model_obs_dim = model.observation_space.shape[0]
    env_obs_dim = obs.shape[0]

    print(f"✅ Training env observation dim: {env_obs_dim}")
    print(f"✅ Model observation space dim: {model_obs_dim}")

    if env_obs_dim != model_obs_dim:
        print("❌ CRITICAL: Observation dimension mismatch!")

    # Save environment specifications for verification
    env_specs = {
        "observation_dim": env_obs_dim,
        "num_flows": train_env.num_flows,
        "candidates_per_flow": train_env.candidates_per_flow,
        "num_edges": train_env.num_edges,
        "graph_nodes": len(G.nodes()),
        "graph_edges": len(G.edges())
    }

    with open(models_dir / "env_specs.json", "w") as f:
        json.dump(env_specs, f, indent=2)
    print(f"✅ Environment specs saved to: {models_dir / 'env_specs.json'}")
    # === END ADDED CODE ===

    model_filename = "rl_PPO.zip"
    model.save(models_dir / model_filename)
    print(f"✅ Model saved to: {models_dir / model_filename}")

    # --- Evaluation, Visualization, Animation, and PDF Report ---

    print("\nEvaluating trained model and visualizing routing decisions...\n")
    obs, _ = eval_env.reset()
    done = False
    step = 0
    max_steps = cfg.get("visualization", {}).get("frames", 1)

    saved_images = []
    actions_over_time = []
    flow_metrics_over_time = []

    if True:  # or just run once
        action, _ = model.predict(obs, deterministic=True)
        filename = visualize_routing(eval_env, action, step=0)

        # Collect per-flow metrics for display
        flow_metrics = []
        for flow_idx in range(eval_env.num_flows):
            info = eval_env.last_info if hasattr(eval_env, 'last_info') else {}
            latency = info.get("latency", 0) if info else 0
            throughput = info.get("throughput", 0) if info else 0
            flow_metrics.append({"latency": latency, "throughput": throughput})

        # Visualize and save plot; get filename
        filename = visualize_routing(
            eval_env, action, flow_metrics=flow_metrics, step=step)
        saved_images.append(filename)
        actions_over_time.append(action)
        flow_metrics_over_time.append(flow_metrics)

        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_env.last_info = info  # Store last info for metrics

        done = terminated or truncated
        step += 1
        time.sleep(0.1)  # optional delay for stability

    # Create GIF animation from saved images
    gif_path = animate_routing_over_time(
        eval_env, actions_over_time, flow_metrics_over_time)

    # Create PDF report aggregating all visualizations
    pdf_path = create_pdf_report(saved_images)

    print(
        f"\nEvaluation complete.\nAnimation GIF saved to: {gif_path}\nPDF report saved to: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
