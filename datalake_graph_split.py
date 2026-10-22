"""
datalake_graph_split.py — Split an entity relationship graph into two readable figures

The pipeline scripts (datalake_rule_based_pipeline.py, datalake_nlp_pipeline.py,
datalake_llm_pipeline.py) each render one entity relationship graph using
nx.spring_layout over ALL nodes, including files with no customer/complaint link.
On a 55-file run, isolated nodes crowd the layout and visually drown out the
genuine connections, which are the part worth showing clearly.

This script rebuilds the same graph from a pipeline's SQLite DB and renders it
as two separate figures instead of one:
  1. <prefix>_connected_<ts>.png — only nodes with at least one edge (degree >= 1),
     laid out with spring_layout exactly as in the original pipeline.
  2. <prefix>_isolated_<ts>.png  — only nodes with no edges, arranged in a plain
     grid (a spring/circular layout makes no sense with zero edges to optimise for).

Usage:
  python datalake_graph_split.py output/datalake_llm_phi3_mini_20260615_215223.db
  python datalake_graph_split.py output/datalake_llm_phi3_mini_20260615_215223.db --title "LLM Pipeline (phi3:mini)"
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

NODE_COLORS = {
    "Customer":                        "#4472C4",
    "Complaint":                       "#ED7D31",
    "DataFile_identities":             "#70AD47",
    "DataFile_service_requests":       "#FF0000",
    "DataFile_financial_transactions": "#FFD966",
    "DataFile_unknown":                "#BFBFBF",
}


def clean_customer_label(cid: str, name: str) -> str:
    """Return a readable node label, falling back to customer_id when name is blank.

    Some LLM-pipeline runs leave customer_id holding the model's malformed output
    (the literal prompt placeholder, a sentence, or a comma-separated number list)
    instead of a real identifier. Those are not useful as figure labels, so they
    are shown as a generic marker rather than the raw garbled text.
    """
    if name:
        return name
    looks_malformed = (
        "<" in cid or ">" in cid or "|" in cid or "," in cid
        or cid.count(" ") > 2 or len(cid) > 24
    )
    if looks_malformed:
        return "Unresolved ID"
    return cid


def build_graph(db_path: Path) -> nx.DiGraph:
    G = nx.DiGraph()
    with sqlite3.connect(db_path) as conn:
        for cid, name in conn.execute("SELECT customer_id, name FROM customers"):
            label = clean_customer_label(cid, name)
            G.add_node(f"cust:{cid}", label=label, node_type="Customer")

        for comp_id, cid in conn.execute(
            "SELECT complaint_id, customer_id FROM customer_complaints"
        ):
            G.add_node(f"comp:{comp_id}", label=comp_id[:14], node_type="Complaint")
            if cid:
                G.add_edge(f"cust:{cid}", f"comp:{comp_id}", relation="HAS_COMPLAINT")

        for fid, fname, cat, sec, cid in conn.execute(
            "SELECT file_id, file_name, info_type, security_class, customer_id FROM data_files"
        ):
            G.add_node(f"file:{fid}", label=fname[:22], node_type=f"DataFile_{cat}", security=sec)
            if cid:
                G.add_edge(f"cust:{cid}", f"file:{fid}", relation="RELATED_TO")
    return G


def legend_patches():
    return [
        mpatches.Patch(color=c, label=lbl.replace("DataFile_", "File - ").replace("_", " "))
        for lbl, c in NODE_COLORS.items()
    ]


def save_connected(G: nx.DiGraph, graph_path: Path, title: str) -> int:
    connected_nodes = [n for n in G.nodes if G.degree(n) > 0]
    H = G.subgraph(connected_nodes).copy()
    if H.number_of_nodes() == 0:
        print("No connected nodes - skipping connected figure.")
        return 0

    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(H, seed=42, k=1.6)
    color_map = [NODE_COLORS.get(H.nodes[n].get("node_type", ""), "#BFBFBF") for n in H.nodes]
    labels    = {n: H.nodes[n].get("label", n) for n in H.nodes}

    nx.draw_networkx_nodes(H, pos, node_color=color_map, node_size=2200, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, ax=ax, font_weight="bold")
    nx.draw_networkx_edges(
        H, pos, ax=ax, arrows=True, arrowsize=22,
        edge_color="#444444", width=1.8, connectionstyle="arc3,rad=0.12",
    )
    edge_labels = {(u, v): d["relation"] for u, v, d in H.edges(data=True)}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.legend(handles=legend_patches(), loc="upper left", fontsize=8)
    ax.set_title(f"{title} - Connected Objects ({H.number_of_nodes()} nodes)", fontsize=13, pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Connected figure saved -> {graph_path} ({H.number_of_nodes()} nodes, {H.number_of_edges()} edges)")
    return H.number_of_nodes()


def save_isolated(G: nx.DiGraph, graph_path: Path, title: str, n_cols: int = 8) -> int:
    isolated_nodes = [n for n in G.nodes if G.degree(n) == 0]
    if not isolated_nodes:
        print("No isolated nodes - skipping isolated figure.")
        return 0

    n_rows = (len(isolated_nodes) + n_cols - 1) // n_cols
    pos = {n: (i % n_cols, -(i // n_cols)) for i, n in enumerate(isolated_nodes)}
    H = G.subgraph(isolated_nodes)

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.6), max(4, n_rows * 1.4)))
    color_map = [NODE_COLORS.get(H.nodes[n].get("node_type", ""), "#BFBFBF") for n in H.nodes]
    labels    = {n: H.nodes[n].get("label", n) for n in H.nodes}

    nx.draw_networkx_nodes(H, pos, node_color=color_map, node_size=1800, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=7, ax=ax, font_weight="bold")

    ax.legend(handles=legend_patches(), loc="upper center", bbox_to_anchor=(0.5, 1.08),
              ncol=3, fontsize=8)
    ax.set_title(f"{title} - Unconnected Objects ({len(isolated_nodes)} nodes)", fontsize=13, pad=30)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Isolated figure saved -> {graph_path} ({len(isolated_nodes)} nodes)")
    return len(isolated_nodes)


def save_combined(G: nx.DiGraph, graph_path: Path, title: str) -> None:
    """Single-figure render of the full graph (all nodes), matching the original
    pipeline-script style. Useful for re-rendering a pipeline's graph after a
    label fix without re-running the whole pipeline."""
    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(G, seed=42, k=2.8)
    color_map = [NODE_COLORS.get(G.nodes[n].get("node_type", ""), "#BFBFBF") for n in G.nodes]
    labels    = {n: G.nodes[n].get("label", n) for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2000, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True, arrowsize=22,
        edge_color="#444444", width=1.8, connectionstyle="arc3,rad=0.12",
    )
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    ax.legend(handles=legend_patches(), loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=13, pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined figure saved -> {graph_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def main():
    parser = argparse.ArgumentParser(description="Render a pipeline's entity graph, split or combined")
    parser.add_argument("db_path", type=Path, help="Path to the pipeline's .db file")
    parser.add_argument("--title", default=None, help="Title prefix for the figure(s) (default: derived from db filename)")
    parser.add_argument("--out-prefix", default=None, help="Output filename prefix (default: derived from db filename)")
    parser.add_argument("--mode", choices=["split", "combined"], default="split",
                         help="'split' (default) produces connected+isolated figures; "
                              "'combined' reproduces the original single-figure pipeline style "
                              "(useful for re-rendering after a code fix without re-running the pipeline)")
    args = parser.parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"DB not found: {args.db_path}")

    stem = args.db_path.stem.replace("datalake_", "")
    title = args.title or stem.replace("_", " ").title()
    prefix = args.out_prefix or stem

    G = build_graph(args.db_path)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if args.mode == "combined":
        combined_path = OUTPUT_DIR / f"graph_{prefix}_{TIMESTAMP}.png"
        save_combined(G, combined_path, title)
    else:
        connected_path = OUTPUT_DIR / f"graph_{prefix}_connected_{TIMESTAMP}.png"
        isolated_path  = OUTPUT_DIR / f"graph_{prefix}_isolated_{TIMESTAMP}.png"
        save_connected(G, connected_path, title)
        save_isolated(G, isolated_path, title)


if __name__ == "__main__":
    main()
