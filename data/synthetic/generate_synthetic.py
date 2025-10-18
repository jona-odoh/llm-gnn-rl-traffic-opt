import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random

def generate_topology(num_nodes: int, edge_prob: float, seed: int):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and rng.random() < edge_prob:
                capacity = int(rng.integers(50, 200))
                edges.append({"src": i, "dst": j, "capacity": capacity})
    if len(edges) == 0:
        edges.append({"src": 0, "dst": 1, "capacity": 100})
    return edges

def generate_traffic_timeseries(edges, timesteps: int, seed: int):
    rng = np.random.default_rng(seed)
    records = []
    for t in range(timesteps):
        for e in edges:
            base = rng.uniform(10, 100)
            seasonal = 20 * np.sin(2 * np.pi * t / 50)
            noise = rng.normal(0, 5)
            bytes_ = max(1, base + seasonal + noise)
            records.append({
                "time": t,
                "src": e["src"],
                "dst": e["dst"],
                "bytes": float(bytes_)
            })
    return records

def generate_logs(timesteps: int, num_lines: int, seed: int):
    rng = random.Random(seed)
    protos = ["TCP", "UDP", "ICMP", "DENIED", "ALERT"]
    lines = []
    for t in range(timesteps):
        for _ in range(num_lines):
            p = rng.choice(protos)
            lines.append(json.dumps({
                "time": t,
                "raw": f"{p} flow src=10.0.0.{rng.randint(1,50)} dst=10.0.1.{rng.randint(1,50)} bytes={rng.randint(40,5000)}"
            }))
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--edge-prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = generate_topology(args.num_nodes, args.edge_prob, args.seed)
    traffic = generate_traffic_timeseries(edges, args.timesteps, args.seed)
    logs = generate_logs(args.timesteps, num_lines=3, seed=args.seed)

    pd.DataFrame(edges).to_csv(out_dir / "topology_edges.csv", index=False)
    pd.DataFrame(traffic).to_parquet(out_dir / "traffic_timeseries.parquet")
    with open(out_dir / "logs.jsonl", "w") as f:
        for l in logs:
            f.write(l + "\n")

    print("Synthetic data generated.")

if __name__ == "__main__":
    main()