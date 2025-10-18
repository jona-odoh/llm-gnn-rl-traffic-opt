from typing import List, Tuple, Optional, Dict
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path


class TrafficGraphDataset(InMemoryDataset):
    """
    Builds time-stepped graph samples for supervised learning:
    Each sample: features derived from previous window_size steps -> predict next load.
    Edge-focused regression.
    """

    def __init__(
        self,
        root: str,
        traffic_file: str,
        topology_file: str,
        window_size: int = 4,
        horizon: int = 1,
        transform=None,
        pre_transform=None
    ):
        self.traffic_file = traffic_file
        self.topology_file = topology_file
        self.window_size = window_size
        self.horizon = horizon
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.traffic_file, self.topology_file]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        # Assume data already present.
        pass

    def process(self):
        traffic_df = pd.read_parquet(self.traffic_file)
        topo_df = pd.read_csv(self.topology_file)

        # Build base graph
        G = nx.from_pandas_edgelist(
            topo_df, "src", "dst", edge_attr=True, create_using=nx.DiGraph())

        # Suppose traffic_df columns: time, src, dst, bytes
        # Build edge time series keyed by (src, dst)
        grouped = traffic_df.groupby(["src", "dst"])
        edge_series = {}
        for (s, d), grp in grouped:
            edge_series[(s, d)] = grp.sort_values("time")["bytes"].values

        # Align lengths (truncate to min)
        min_len = min(len(v) for v in edge_series.values())
        for k in edge_series:
            edge_series[k] = edge_series[k][:min_len]

        edge_keys = list(edge_series.keys())
        series_matrix = torch.tensor(
            [edge_series[k] for k in edge_keys], dtype=torch.float)  # [E, T]

        samples = []
        # time major windows
        for t in range(self.window_size, min_len - self.horizon):
            past_window = series_matrix[:, t - self.window_size:t]  # [E, W]
            target = series_matrix[:, t + self.horizon - 1]  # [E]
            # Node features (simplified): degree or random placeholder
            node_feats = []
            for node in G.nodes():
                deg = G.out_degree(node) + G.in_degree(node)
                node_feats.append([deg])
            x = torch.tensor(node_feats, dtype=torch.float)  # [N, 1]

            # Edge index mapping
            node_to_idx = {n: i for i, n in enumerate(G.nodes())}
            edge_index = torch.tensor([[node_to_idx[s] for (s, d) in edge_keys],
                                       [node_to_idx[d] for (s, d) in edge_keys]], dtype=torch.long)
            # Edge features = window history statistics
            mean = past_window.mean(dim=1, keepdim=True)
            std = past_window.std(dim=1, keepdim=True).clamp(min=1e-6)
            last = past_window[:, -1].unsqueeze(-1)
            edge_attr = torch.cat([mean, std, last], dim=1)  # shape [E, 3]

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=target.unsqueeze(-1)  # [E, 1]
            )
            samples.append(data)

        data, slices = self.collate(samples)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
