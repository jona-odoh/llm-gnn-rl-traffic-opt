import argparse
import yaml
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from pathlib import Path
from .dataset import TrafficGraphDataset
from .model import EdgeRegressionGNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm
import random
import os
import json

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(pred, batch.y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        preds.append(out.cpu())
        targets.append(batch.y.cpu())
    preds = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    return mse, mae

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    data_dir = cfg["paths"]["data_dir"]
    traffic_file = f"{data_dir}/traffic_timeseries.parquet"
    topo_file = f"{data_dir}/topology_edges.csv"

    dataset = TrafficGraphDataset(
        root="data/processed/gnn_cache",
        traffic_file=traffic_file,
        topology_file=topo_file,
        window_size=cfg["gnn"]["window_size"],
        horizon=cfg["gnn"]["prediction_horizon"]
    )
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len],
                                                generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = DataLoader(train_set, batch_size=cfg["gnn"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["gnn"]["batch_size"])
    test_loader = DataLoader(test_set, batch_size=cfg["gnn"]["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EdgeRegressionGNN(
        node_in=1,
        edge_in=3,
        hidden=cfg["gnn"]["hidden_dim"],
        out_dim=cfg["gnn"]["output_dim"],
        num_layers=cfg["gnn"]["num_layers"],
        dropout=cfg["gnn"]["dropout"]
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["gnn"]["lr"], weight_decay=cfg["gnn"]["weight_decay"])

    best_val = float("inf")
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, cfg["gnn"]["epochs"] + 1):
        tr_loss = train_epoch(model, train_loader, optim, device)
        val_mse, val_mae = eval_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_mse": val_mse, "val_mae": val_mae})
        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} val_mse={val_mse:.4f} val_mae={val_mae:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), models_dir / "best_gnn.pt")

    # Test
    model.load_state_dict(torch.load(models_dir / "best_gnn.pt"))
    test_mse, test_mae = eval_epoch(model, test_loader, device)
    print(f"Test MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")
    with open(reports_dir / "gnn_metrics.json", "w") as f:
        json.dump({"test_mse": test_mse, "test_mae": test_mae, "history": history}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)