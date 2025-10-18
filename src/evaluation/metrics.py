import numpy as np
from typing import Dict, List


def compute_prediction_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = ((y_true - y_pred) ** 2).mean()
    mae = np.abs(y_true - y_pred).mean()
    return {"mse": mse, "mae": mae}


def aggregate_rl_metrics(records: List[dict]) -> Dict[str, float]:
    latency = np.mean([r["latency"] for r in records])
    throughput = np.mean([r["throughput"] for r in records])
    packet_loss = np.mean([r["packet_loss"] for r in records])
    reward = np.mean([r["reward"] for r in records])
    return {
        "avg_latency": latency,
        "avg_throughput": throughput,
        "avg_packet_loss": packet_loss,
        "avg_reward": reward
    }
