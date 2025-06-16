#!/usr/bin/env python3
"""
generate_dag.py

Generate a random, connected Directed Acyclic Graph (DAG) with:
  - a specified number of “input” nodes (each in‐degree = 0),
  - a specified number of “output” nodes (each out‐degree = 0),
  - a target average degree per node,
  - a total node count,
  - and a (uniformly random) processing‐time for each node.

Writes:
  1) A plain‐text file (default: dag_output.txt) in this format:

       # Nodes
       node_id  processing_time
       ...
       # Edges
       src_id  dst_id
       ...

  2) A “connected”‐DAG visualization (PNG) arranged in layers (inputs at the top, outputs at the bottom),
     colored by node‐type.  By default, it names the PNG exactly as the text file but with “.png”.

Usage:
    python generate_dag.py
        --num_inputs K_IN
        --num_outputs K_OUT
        --avg_degree AVG_DEG
        --total_nodes N
        --min_proc_time MIN_PT
        --max_proc_time MAX_PT
        [--output_file OUTPUT_FILENAME]

Example:
    python generate_dag.py \
      --num_inputs 2 \
      --num_outputs 1 \
      --avg_degree 2.0 \
      --total_nodes 7 \
      --min_proc_time 5 \
      --max_proc_time 20 \
      --output_file mygraph.txt
"""

import argparse
from pathlib import Path
import random
import sys

# We only import these when we actually do the drawing.
import networkx as nx
import matplotlib.pyplot as plt


DIMENSIONS = (10, 10)
FILE_STORAGE_PATH = "graphs"

def compute_possible_edge_count(n: int, k_in: int, k_out: int) -> int:
    """
    Count how many (i → j) pairs are allowed under our DAG constraints:
      • i < j  (to avoid cycles)
      • i is NOT one of the last k_out nodes (so i may have outgoing edges)
      • j is NOT one of the first k_in nodes (so j may have incoming edges)

    Equivalently:
      i ∈ [0 .. (n - k_out - 1)]
      j ∈ [max(i+1, k_in) .. (n - 1)]
    """
    count = 0
    for i in range(0, n - k_out):
        j_start = max(i + 1, k_in)
        if j_start <= n - 1:
            count += (n - j_start)
    return count


def generate_random_connected_dag(
    n: int,
    k_in: int,
    k_out: int,
    avg_deg: float,
    min_pt: int,
    max_pt: int,
) -> (dict, list):
    """
    Generates a *connected* DAG with:
      - n total nodes (0..n-1)
      - k_in input nodes  = {0..k_in-1}, each with in-degree = 0
      - k_out output nodes = {n-k_out..n-1}, each with out-degree = 0
      - directed edges only from smaller index → larger index
      - at least one undirected path connecting every node (“weak” connectivity)
      - an expected total number of edges ≈ avg_deg * n
      - processing time for each node drawn uniformly from [min_pt..max_pt].

    Returns:
      proc_times: dict { node_id → processing_time }
      edges:      list of (src, dst) pairs
    """

    if k_in + k_out > n:
        raise ValueError("num_inputs + num_outputs must be ≤ total_nodes")

    # 1) Assign a random processing time (integer) to each node:
    proc_times = { node_id: random.randint(min_pt, max_pt) for node_id in range(n) }

    # 2) We will first force a “spanning‐tree” of edges under our DAG constraints, to ensure connectivity.
    edge_set = set()

    # Decide on a single “hub” node j0 that all inputs will point to; j0 must NOT be an input.
    # Prefer an internal node if one exists; otherwise use the first output node.
    if k_in < n - k_out:
        j0 = k_in
    else:
        # No internals exist ⇒ the first output is at index n-k_out
        j0 = n - k_out

    # Connect each input i ∈ [0..k_in-1] → j0.
    # This ties all inputs (roots) together in the same weak component via j0.
    for i in range(0, k_in):
        edge_set.add((i, j0))

    # Now, for every other node v ∈ [k_in .. n-1], v ≠ j0, ensure it has at least one parent.
    # Choose a random u ∈ [0..min(v-1, n-k_out-1)] so that u < v and u is not an output.
    for v in range(k_in, n):
        if v == j0:
            continue
        max_u = min(v - 1, n - k_out - 1)
        # max_u must be ≥ 0 because:
        #   • v ≥ k_in ≥ 1 ⇒ v-1 ≥ 0
        #   • n-k_out-1 ≥ k_in-1 ≥ 0 (since k_in + k_out ≤ n)
        u = random.randint(0, max_u)
        edge_set.add((u, v))

    # At this point, we have exactly one “incoming edge” for each non-input node (including j0),
    # and each input has an outgoing to j0.  The resulting underlying undirected graph is connected.

    current_edge_count = len(edge_set)
    desired_edge_count = avg_deg * n
    max_possible_edges = compute_possible_edge_count(n, k_in, k_out)

    # If the desired total is less than current_edge_count, we cannot delete edges
    # (we need all of them for connectivity).  So we warn, and we won’t add any more edges.
    if desired_edge_count < current_edge_count:
        print(
            f"Warning: Requested avg_degree={avg_deg:.2f} ⇒ desired edges={desired_edge_count:.1f},\n"
            f"  but minimum edges needed for connectivity = {current_edge_count}.\n"
            f"  Final average degree will be {current_edge_count / n:.2f} instead.",
            file=sys.stderr,
        )
        p = 0.0
    else:
        # We want “desired_edge_count - current_edge_count” more edges in expectation.
        remaining_target = desired_edge_count - current_edge_count
        remaining_possible = max_possible_edges - current_edge_count

        if remaining_possible <= 0:
            # No additional slots exist; all allowable pairs are used in edge_set
            p = 0.0
        else:
            p = remaining_target / remaining_possible
            if p > 1.0:
                print(
                    f"Warning: Desired average degree {avg_deg:.2f} is too large for\n"
                    f"  {n} nodes with {k_in} inputs and {k_out} outputs.\n"
                    f"  Clamping additional‐edge probability to 1.0 (all remaining slots).",
                    file=sys.stderr,
                )
                p = 1.0

    # 3) Now sample each remaining “allowed” pair (i < j) not already in edge_set with probability p
    for i in range(0, n - k_out):
        j_start = max(i + 1, k_in)
        for j in range(j_start, n):
            if (i, j) in edge_set:
                continue
            if random.random() < p:
                edge_set.add((i, j))

    # Convert to a sorted list of edges (sorted by src, then dst) for consistency
    edges = sorted(edge_set, key=lambda e: (e[0], e[1]))
    return proc_times, edges


def write_graph_to_file(
    filename: str,
    proc_times: dict,
    edges: list,
    dimensions: tuple[int, int] = DIMENSIONS
) -> None:
    """
    Writes out:
      # Nodes
      node_id  processing_time
      ...
      # Edges
      src_id  dst_id
      ...
    """
    parent_dir = Path(FILE_STORAGE_PATH) / filename
    parent_dir.mkdir(exist_ok=True, parents=True)

    with open(str(Path(FILE_STORAGE_PATH) / filename / (filename + "_input.txt")), "w") as f:
        f.write("# Nodes\n")
        for node_id in sorted(proc_times.keys()):
            f.write(f"{node_id} {proc_times[node_id]}\n")

        f.write("# Edges\n")
        for (src, dst) in edges:
            f.write(f"{src} {dst}\n")

        f.write("# Dimensions\n")
        f.write(f"{dimensions[0]} {dimensions[1]}\n")


def visualize_and_save(
    proc_times: dict,
    edges: list,
    output_filename: str,
    k_in: int,
    k_out: int,
    dimensions: tuple[int, int] = DIMENSIONS
) -> None:
    """
    Creates a layered DAG visualization with:
      - Inputs (0..k_in-1) at the top (colored light blue)
      - Internal nodes (k_in..n-k_out-1) in the middle (light green)
      - Outputs (n-k_out..n-1) at the bottom (light coral)

    It uses networkx.multipartite_layout on the “layer” attribute so that data flows
    top→down.  Saves to OUTPUT_FILENAME with “.png”.
    """

    n = len(proc_times)
    G = nx.DiGraph()
    G.add_nodes_from(proc_times.keys())
    G.add_edges_from(edges)

    # 1) Compute “layer” (longest distance from any input) for each node via a DP over topological order:
    topo = list(nx.topological_sort(G))
    layer = {}
    # Inputs have layer = 0
    for i in range(k_in):
        layer[i] = 0

    for v in topo:
        if v < k_in:
            continue  # already layer=0
        preds = list(G.predecessors(v))
        if not preds:
            # This can only happen if v == j0 and k_in==0, but k_in≥1 by design
            layer[v] = 0
        else:
            layer[v] = max(layer[u] + 1 for u in preds)

    # Assign layer as a node attribute
    nx.set_node_attributes(G, layer, "layer")

    # 2) Use multipartite_layout for layering, then invert y-axis so layer=0 is at the top
    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")

    # Invert y to get inputs at the top (y=0), outputs at bottom (y negative).
    for node_id, (x, y) in pos.items():
        pos[node_id] = (x, -y)

    # 3) Choose colors by node‐type
    colors = []
    for node_id in G.nodes():
        if node_id < k_in:
            colors.append("lightblue")
        elif node_id >= n - k_out:
            colors.append("lightcoral")
        else:
            colors.append("lightgreen")

    # 4) Draw
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=700,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
        width=1.5,
    )
    # Label by node ID only.  (If you prefer processing_time in the label, replace with f"{node}\n{proc_times[node]}")
    labels = {node_id: str(node_id) for node_id in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=10,
        font_color="black",
    )

    plt.title("Connected DAG\n(inputs: light blue → outputs: light coral)")
    plt.axis("off")
    plt.tight_layout()

    png_name = output_filename.rsplit(".", 1)[0] + ".png"
    path = "graphs/" + output_filename + "/" + output_filename + ".png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Visualization saved as '{path}'.")


def parse_arguments():
    """
    Command-line interface
    ----------------------
    Positional arguments (enter them in this order):

      1. num_inputs     – int  – number of input nodes (in-degree 0)
      2. num_outputs    – int  – number of output nodes (out-degree 0)
      3. avg_degree     – float– target average degree per node
      4. total_nodes    – int  – total number of nodes in the DAG
      5. min_proc_time  – int  – minimum processing time (inclusive)
      6. max_proc_time  – int  – maximum processing time (inclusive)

    Optional positional argument:

      7. output_file    – str  – path to output file
                                (default: dag_output.txt)

    Example
    -------
    $ python generate_dag.py 3 2 1.8 50 1 5 my_dag.txt
    """
    parser = argparse.ArgumentParser(
        prog="generate_dag.py",
        description=(
            "Generate a connected random DAG with the given parameters and "
            "write it to a file, plus a layered PNG visualization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # required positional parameters (no flags needed)
    parser.add_argument("num_inputs",   type=int,   help="Number of input nodes.")
    parser.add_argument("num_outputs",  type=int,   help="Number of output nodes.")
    parser.add_argument("avg_degree",   type=float, help="Target average degree per node.")
    parser.add_argument("total_nodes",  type=int,   help="Total number of nodes in the DAG.")
    parser.add_argument("min_proc_time", type=int,  help="Minimum per-node processing time.")
    parser.add_argument("max_proc_time", type=int,  help="Maximum per-node processing time.")

    # optional final positional parameter (nargs='?' → 0 or 1 values accepted)
    parser.add_argument(
        "output_file",
        nargs="?",
        default="dag_output",
        help="Output file path (default: dag_output.txt)",
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    n = args.total_nodes
    k_in = args.num_inputs
    k_out = args.num_outputs
    avg_deg = args.avg_degree
    min_pt = args.min_proc_time
    max_pt = args.max_proc_time
    out_file = args.output_file

    # Sanity checks
    if n <= 0:
        print("Error: total_nodes must be a positive integer.", file=sys.stderr)
        sys.exit(1)
    if k_in < 1 or k_out < 1:
        print("Error: At least one input node and one output node are required.", file=sys.stderr)
        sys.exit(1)
    if k_in + k_out > n:
        print("Error: num_inputs + num_outputs must be ≤ total_nodes.", file=sys.stderr)
        sys.exit(1)
    if min_pt < 0 or max_pt < 0 or min_pt > max_pt:
        print(
            "Error: Invalid processing time range. Ensure 0 ≤ min_proc_time ≤ max_proc_time.",
            file=sys.stderr,
        )
        sys.exit(1)

    proc_times, edges = generate_random_connected_dag(n, k_in, k_out, avg_deg, min_pt, max_pt)
    write_graph_to_file(out_file, proc_times, edges)

    print(f"DAG successfully written to '{out_file}'.")
    print(f"  • Nodes = {n}  (inputs = {k_in}, outputs = {k_out})")
    print(f"  • Edges = {len(edges)}  (expected ≈ {avg_deg * n:.1f} )")
    print(f"  • Processing times ∈ [{min_pt}, {max_pt}]")

    # Draw and save a layered visualization
    visualize_and_save(proc_times, edges, out_file, k_in, k_out)


if __name__ == "__main__":
    main()
