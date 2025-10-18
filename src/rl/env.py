import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, Optional


class RoutingEnv(gym.Env):
    """
    Simplified routing optimization environment:
    - Graph with capacities and predicted loads (provided externally or random).
    - Action: For k selected flows, choose route index among candidate paths (discrete).
    - State: concatenation of normalized edge utilizations + simple global stats.
    Reward encourages low latency and high throughput, penalizes congestion.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        graph: nx.DiGraph,
        num_flows: int = 5,
        candidates_per_flow: int = 3,
        max_queue: int = 10000,
        latency_base: float = 2.0,
        congestion_threshold: float = 0.8,
        penalty_congestion: float = 5.0,
        reward_weights: Dict[str, float] = None,
        seed: int = 42
    ):
        super().__init__()
        self.graph = graph
        self.edges = list(self.graph.edges())
        self.num_edges = len(self.edges)
        self.num_flows = num_flows
        self.candidates_per_flow = candidates_per_flow
        self.max_queue = max_queue
        self.latency_base = latency_base
        self.congestion_threshold = congestion_threshold
        self.penalty_congestion = penalty_congestion
        self.reward_weights = reward_weights or {
            "latency": 0.5, "throughput": 0.3, "loss": 0.2}
        self.rng = np.random.default_rng(seed)
        # Precompute candidate paths for flows (random src-dst)
        self.flows = self._generate_flows()
        self.path_candidates = self._compute_path_candidates()

        # Observation: edge utilization + mean utilization + variance
        self.observation_space = spaces.Box(
            low=0, high=2.0, shape=(self.num_edges + 2,), dtype=np.float32)
        # Action: choose one path index per flow => MultiDiscrete
        self.action_space = spaces.MultiDiscrete(
            [self.candidates_per_flow] * self.num_flows)

        self.reset(seed=seed)

    def _generate_flows(self):
        nodes = list(self.graph.nodes())
        flows = []
        for _ in range(self.num_flows):
            s, d = self.rng.choice(nodes, 2, replace=False)
            flows.append((s, d))
        return flows

    def _compute_path_candidates(self):
        candidates = {}
        for idx, (s, d) in enumerate(self.flows):
            all_paths = list(nx.all_simple_paths(self.graph, s, d, cutoff=5))
            if len(all_paths) == 0:
                all_paths = [[s, d]]  # fallback
            self.rng.shuffle(all_paths)
            trimmed = all_paths[:self.candidates_per_flow]

            # Pad if fewer paths than candidates_per_flow
            while len(trimmed) < self.candidates_per_flow:
                trimmed.append(trimmed[-1])  # Repeat last path

            candidates[idx] = trimmed
        return candidates

    def _initial_utilization(self):
        util = self.rng.uniform(0.1, 0.5, size=self.num_edges)
        return util

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.edge_utilization = self._initial_utilization()
        self.time = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        mean_u = self.edge_utilization.mean()
        var_u = self.edge_utilization.var()
        return np.concatenate([self.edge_utilization, [mean_u, var_u]]).astype(np.float32)

    def step(self, action):
        # action: path indices for each flow
        assert len(action) == self.num_flows
        # Simulate traffic assignment
        added_load = np.zeros(self.num_edges)
        throughput = 0.0
        packet_loss = 0.0
        latency = 0.0

        for flow_idx, path_choice in enumerate(action):
            if path_choice >= len(self.path_candidates[flow_idx]):
                raise IndexError(
                    f"Invalid path_choice {path_choice} for flow_idx {flow_idx}, only {len(self.path_candidates[flow_idx])} paths available.")
            path = self.path_candidates[flow_idx][path_choice]
            edges_in_path = list(zip(path[:-1], path[1:]))
            path_load = self.rng.uniform(5.0, 20.0)  # synthetic demand
            throughput += path_load
            for e_idx, e in enumerate(self.edges):
                if e in edges_in_path:
                    added_load[e_idx] += path_load

        # Update utilization: capacity from graph attribute 'capacity' or default
        capacities = np.array([self.graph.get_edge_data(u, v).get(
            "capacity", 100.0) for (u, v) in self.edges])
        self.edge_utilization += added_load / capacities
        # Clip
        self.edge_utilization = np.clip(self.edge_utilization, 0.0, 2.0)

        # Compute metrics
        congested = self.edge_utilization > self.congestion_threshold
        congestion_penalty = congested.sum() * self.penalty_congestion
        latency = self.latency_base + (self.edge_utilization.mean() * 10)
        packet_loss = congested.sum() / self.num_edges

        reward = (
            -self.reward_weights["latency"] * latency +
            self.reward_weights["throughput"] * throughput -
            self.reward_weights["loss"] * packet_loss -
            congestion_penalty * 0.01
        )

        self.time += 1
        terminated = False
        truncated = self.time >= 100
        obs = self._get_obs()
        info = {
            "latency": latency,
            "throughput": throughput,
            "packet_loss": packet_loss,
            "congestion_edges": int(congested.sum()),
            "reward": reward
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Time {self.time} mean_util={self.edge_utilization.mean():.3f}")
