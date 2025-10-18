import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import imageio
from fpdf import FPDF


def visualize_routing(env: 'RoutingEnv', action: np.ndarray, flow_metrics=None, step=None):
    """
    Visualize the network graph with chosen paths highlighted and edge utilizations shown.

    Args:
        env: Your RoutingEnv instance.
        action: Array of path choices per flow (same as RL action).
        flow_metrics: List of dicts with keys like 'latency' and 'throughput' per flow (optional).
        step: Current evaluation step for display (optional).
    """

    G = env.graph
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes

    plt.figure(figsize=(12, 8))

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
    nx.draw_networkx_labels(G, pos)

    # Base edges - light gray, thin lines
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(),
                           edge_color='gray', width=1, alpha=0.4)

    # Highlight edges used in chosen paths
    edges_in_paths = []
    for flow_idx, path_choice in enumerate(action):
        path = env.path_candidates[flow_idx][path_choice]
        path_edges = list(zip(path[:-1], path[1:]))
        edges_in_paths.extend(path_edges)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_in_paths,
        edge_color='red',
        width=3,
        alpha=0.8,
        label='Chosen paths'
    )

    # Visualize edge utilization as edge widths and colors
    utilizations = env.edge_utilization
    norm_util = (utilizations - utilizations.min()) / \
        (np.ptp(utilizations) + 1e-6)  # normalize 0-1

    edge_colors = plt.cm.viridis(norm_util)
    edge_widths = 1 + 4 * norm_util  # widths between 1 and 5

    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        label='Edge utilization'
    )

    plt.title(
        f"Network Graph with Chosen Routing Paths and Edge Utilization\nStep {step if step is not None else ''}")
    plt.axis('off')
    plt.legend()

    # Show per-flow metrics text on plot
    if flow_metrics is not None and step is not None:
        metrics_text = "\n".join(
            [f"Flow {i}: Latency={m['latency']:.2f}, Throughput={m['throughput']:.2f}"
             for i, m in enumerate(flow_metrics)]
        )
        plt.gcf().text(0.02, 0.02, metrics_text, fontsize=9,
                       verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.6))

    # Ensure folder exists
    save_dir = Path("reports/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filename with timestamp and step number
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_dir / f"routing_visualization_step{step}_{timestamp}.png"

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Routing visualization saved to: {filename}")
    # Don't auto-open file here; let caller handle if needed

    return filename


def animate_routing_over_time(env, actions_over_time, flow_metrics_over_time, gif_path="reports/visualizations/routing_animation.gif"):
    images = []
    for step, (action, flow_metrics) in enumerate(zip(actions_over_time, flow_metrics_over_time)):
        filename = visualize_routing(
            env, action, flow_metrics=flow_metrics, step=step)
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_path, images, duration=0.5)
    print(f"Animation saved to {gif_path}")
    return gif_path


def create_pdf_report(image_files, pdf_path="reports/visualizations/routing_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for image_file in image_files:
        pdf.add_page()
        pdf.image(str(image_file), x=10, y=10, w=pdf.w - 20)
    pdf.output(pdf_path)
    print(f"PDF report saved to {pdf_path}")
    return pdf_path
