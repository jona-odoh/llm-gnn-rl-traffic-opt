import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gnn_history(history_file: Path, out_path: Path):
    with open(history_file, "r") as f:
        data = json.load(f)
    hist = data["history"]
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_mse = [h["val_mse"] for h in hist]
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_mse, label="Val MSE")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GNN Training")
    plt.savefig(out_path / "gnn_training.png", dpi=150)
    plt.close()

def plot_rl_metrics(metrics_file: Path, out_path: Path):
    with open(metrics_file, "r") as f:
        data = json.load(f)
    steps = [d["steps"] for d in data]
    rewards = [d["avg_reward"] for d in data]
    latency = [d["avg_latency"] for d in data]
    plt.figure(figsize=(8,5))
    plt.plot(steps, rewards, label="Avg Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("RL Reward Progress")
    plt.savefig(out_path / "rl_reward.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(steps, latency, label="Avg Latency", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Latency")
    plt.title("RL Latency Progress")
    plt.savefig(out_path / "rl_latency.png", dpi=150)
    plt.close()