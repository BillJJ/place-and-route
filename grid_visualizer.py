#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  visualize_grid.py   (≥ 2025-06-10 spec)
#
#  Placement-file format
#  ---------------------
#  # Grid
#  R C
#
#  # Tiles (row col comp routeCnt)
#  <r> <c> <computeID | -1> <routeCnt>
#  ...
#
#  # Paths (each edge listed in order given in input)
#  EDGE <edgeID> <numConn>
#  <r> <c> <Dir(U/D/L/R)> <srcType(C/R)> <dstType(C/R)>
#  ...
#
#  Visualisation
#  -------------
#  • neat R × C grid
#  • per-tile glyphs
#        compute: circles
#                 · input-compute   : light-blue
#                 · output-compute  : light-coral
#                 · internal        : light-green
#        route  : squares  (light-gray)
#  • each DAG-edge gets its own colour, arrows show direction
#    (anchors attach to *correct* node in each tile)
#
#  Usage:
#        python visualize_grid.py placement.txt           # show on screen
#        python visualize_grid.py placement.txt -o fig.png  # save PNG
# ---------------------------------------------------------------------------

import argparse
import itertools
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

ROUTE_NODE_PROCESSING_TIME = 2

# ---------------------------------------------------------------------------
#   parse placement file
# ---------------------------------------------------------------------------
def parse_placement(path):
    with open(path) as f:
        lines = [ln.rstrip() for ln in f]

    # --- grid size ---------------------------------------------------------
    try:
        grid_idx = lines.index("# Grid")
        R, C = map(int, lines[grid_idx + 1].split())
    except (ValueError, IndexError):
        raise RuntimeError("Missing '# Grid' section")

    # --- tiles -------------------------------------------------------------
    try:
        tile_idx = lines.index("# Tiles (row col comp routeCnt)")
    except ValueError:
        raise RuntimeError("Missing '# Tiles' section")

    tiles = {}   # (r,c) -> dict
    i = tile_idx + 1
    while i < len(lines) and not lines[i].startswith("#"):
        r, c, comp, rcnt = map(int, lines[i].split())
        tiles[(r, c)] = {"comp": comp, "rcnt": rcnt}
        i += 1

    # --- paths -------------------------------------------------------------
    try:
        path_idx = lines.index("# Paths (each edge listed in order given in input)")
    except ValueError:
        raise RuntimeError("Missing '# Paths' section")

    paths = defaultdict(list)   # edgeID -> [(r,c,Dir,srcT,dstT), ...]
    i = path_idx + 1
    num_paths = int(lines[i])
    i += 1
    while num_paths:
        _, eid, nconn = lines[i].split()
        eid, nconn = int(eid), int(nconn)
        i += 1
        eid = -1
        num_paths -= 1
        for _ in range(nconn):
            r, c, d, st, dt = lines[i].split()
            if eid == -1:
                eid = int(r) * C + int(c)
            paths[eid].append((int(r), int(c), d, st, dt))
            i += 1
    
    # --- processing times --------------------------------------------------
    try:
        processing_idx = lines.index("# Node Positions and Processing times")
    except ValueError:
        raise RuntimeError("Missing '# Node Positions and Processing times' section")
    
    num_nodes = int(lines[processing_idx + 1])
    node_pt = {}
    for idx in range(processing_idx + 2, processing_idx + 2 + num_nodes):
        node, r, c, pt = list(map(int, lines[idx].split(" ")))
        node_pt[(r, c)] = pt

    return R, C, tiles, paths, node_pt


# ---------------------------------------------------------------------------
#   classify compute nodes: input / output / internal
# ---------------------------------------------------------------------------
def classify_compute_nodes(tiles, paths):
    indeg = Counter()
    outdeg = Counter()

    for seq in paths.values():
        for edge in seq:
            # first conn: source tile must hold src compute
            r0, c0, d, t1, t2 = edge

            dirc = (0, 0)
            if d == "U": dirc = (-1, 0)
            elif d == "D": dirc = (1, 0)
            elif d == "R": dirc = (0, 1)
            else: dirc = (0, -1) # dirc = left

            if t1 == "C":
                outdeg[(r0, c0)] += 1 # outdegree only matters if compute node
            if t2 == "C":
                indeg[(r0 + dirc[0], c0 + dirc[1])] += 1


    for (r, c), info in tiles.items():
        if info["comp"] == -1:
            continue
        if indeg[(r, c)] == 0:
            info["kind"] = "in"        # input compute
        elif outdeg[(r, c)] == 0:
            info["kind"] = "out"       # output compute
        else:
            info["kind"] = "mid"
    return tiles


# ---------------------------------------------------------------------------
#   anchor helpers
# ---------------------------------------------------------------------------
DIR2DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
colour_cycle = list(itertools.islice(itertools.cycle(
    list(mcolors.TABLEAU_COLORS.values()) +
    list(mcolors.CSS4_COLORS.values())
), 200))


def anchors_for_tile(r, c, info):
    """Return dict label→(x,y) inside tile:
          C  : compute
          R0 : first route   (if present)
          R1 : second route  (if present)
       Simple layout:
          • if compute & route → C on left, R0 on right
          • if 2 routes       → R0 left, R1 right
          • else centre
    """
    base_x, base_y = c + 0.5, r + 0.5
    delta = 0.18
    out = {}
    comp = info["comp"] != -1
    rcnt = info["rcnt"]

    if comp and rcnt == 0:                # just compute
        out["C"] = (base_x, base_y)
    elif comp and rcnt == 1:              # compute + 1 route
        out["C"] = (base_x - delta, base_y)
        out["R0"] = (base_x + delta, base_y)
    elif not comp and rcnt == 1:          # single route
        out["R0"] = (base_x, base_y)
    elif not comp and rcnt == 2:          # two routes
        out["R0"] = (base_x - delta, base_y)
        out["R1"] = (base_x + delta, base_y)
    return out


# ---------------------------------------------------------------------------
#   draw
# ---------------------------------------------------------------------------
def draw(R, C, tiles, paths, save=None):
    fig, ax = plt.subplots(figsize=(1.2 * C, 1.2 * R))
    ax.set_aspect('equal')
    ax.set_xlim(0, C)
    ax.set_ylim(0, R)
    ax.invert_yaxis()
    ax.axis('off')

    # grid
    for r in range(R + 1):
        ax.add_line(mlines.Line2D([0, C], [r, r], lw=1.0, color='black', zorder=1))
    for c in range(C + 1):
        ax.add_line(mlines.Line2D([c, c], [0, R], lw=1.0, color='black', zorder=1))

    # node glyphs & anchor map
    anchor = {}
    for (r, c), info in tiles.items():
        anchor[(r, c)] = anchors_for_tile(r, c, info)
        # draw compute
        if info["comp"] != -1:
            cx, cy = anchor[(r, c)]["C"]
            col = {"in": "#77b5fe", "out": "#f08080", "mid": "#90ee90"}[info["kind"]]
            circ = patches.Circle((cx, cy), 0.17, fc=col, ec='black', lw=1.1, zorder=3)
            ax.add_patch(circ)
            ax.text(cx, cy, str(info["comp"]), ha='center', va='center',
                    fontsize=7, zorder=4)

        # draw route squares
        for k in range(info["rcnt"]):
            tag = f"R{k}"
            if tag not in anchor[(r, c)]:
                continue
            cx, cy = anchor[(r, c)][tag]
            rect = patches.Rectangle((cx - 0.17, cy - 0.17), 0.34, 0.34,
                                     fc='#d3d3d3', ec='black', lw=1.0, zorder=2)
            ax.add_patch(rect)

    # ------------------------------------------------------------------
    #  PROCESSING-TIME HEATMAP   (replace the old block)
    # ------------------------------------------------------------------
    tile_pt = {}                         # (r,c) -> total processing time
    for (r, c), info in tiles.items():
        p = 0
        if info["comp"] != -1:
            p += node_pt.get((r, c), 0)
        p += info["rcnt"] * ROUTE_NODE_PROCESSING_TIME
        if p > 0:
            tile_pt[(r, c)] = p

    crit_pt  = max(tile_pt.values(), default=0)
    total_pt = sum(tile_pt.values())
    thrpt    = 0 if crit_pt == 0 else 1 / crit_pt

    # --- colour map ---------------------------------------------------
    cmap = cm.get_cmap('RdYlGn_r')      #  green → yellow → red
    norm = mcolors.Normalize(vmin=0, vmax=max(crit_pt, 1))

    # draw heat rectangles *behind* grid lines
    for (r, c), p in tile_pt.items():
        color = cmap(norm(p))
        ax.add_patch(
            patches.Rectangle((c, r), 1, 1,
                            fc=color, ec='none', alpha=0.45, zorder=0)
        )

    # colour-bar legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("tile processing time")

    # tiny text block with totals
    ax.text(-0.05, -0.05,
            # f"Σ tile-time = {total_pt}\n"
            f"bottleneck  = {crit_pt}\n",
            # f"throughput ≈ {thrpt:.3f}",
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=8, family='monospace')


    # ---------------------------------------------------------------
    #  ROUTE-SLOT BOOKKEEPING
    # ---------------------------------------------------------------
    slot_taken = defaultdict(lambda: {"R0": None, "R1": None})
    #          (r,c) -> {"R0": edgeID | None,  "R1": edgeID | None}

    edge_pref  = {}    # edgeID -> preferred slot index (0 or 1)

    # ---------------------------------------------------------------
    #  DRAW EDGES with per-edge, per-tile slot consistency
    # ---------------------------------------------------------------
    for eid, seq in paths.items():
        col   = colour_cycle[eid % len(colour_cycle)]
        pref  = edge_pref.setdefault(eid, eid & 1)      # 0 or 1

        for (r, c, d, st, dt) in seq:
            dr, dc = DIR2DELTA[d]
            nr, nc = r + dr, c + dc

            # ---------- choose FROM anchor --------------------------------
            if st == 'C':
                from_lbl = "C"
            else:                     # need a route slot
                lbl = f"R{pref}"
                # if preferred slot already taken by another edge, flip
                if lbl not in anchor[(r, c)] or slot_taken[(r, c)][lbl] not in (None, eid):
                    lbl = "R1" if lbl == "R0" else "R0"
                slot_taken[(r, c)][lbl] = eid
                from_lbl = lbl

            # ---------- choose TO anchor ----------------------------------
            if dt == 'C':
                to_lbl = "C"
            else:
                lbl = f"R{pref}"
                if lbl not in anchor[(nr, nc)] or slot_taken[(nr, nc)][lbl] not in (None, eid):
                    lbl = "R1" if lbl == "R0" else "R0"
                slot_taken[(nr, nc)][lbl] = eid
                to_lbl = lbl


            # ---------- draw the arrow ------------------------------------
            x0, y0 = anchor[(r, c)][from_lbl]
            x1, y1 = anchor[(nr, nc)][to_lbl]

            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=1.6,
                    color=col,
                    shrinkA=2,
                    shrinkB=2),
                zorder=5)


    if save:
        plt.savefig(save, bbox_inches='tight', dpi=200)
    else:
        plt.show()


# ---------------------------------------------------------------------------
#   main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualise tiled DAG placement")
    ap.add_argument("file", help="placement.txt")
    ap.add_argument("-o", "--out", help="save to PNG instead of showing")
    args = ap.parse_args()

    input_file = str(Path("graphs") / args.file / (args.file + "_placement.txt"))
    output_file = str(Path("graphs") / args.file / (args.file + "_grid.png"))

    R, C, tiles, paths, node_pt = parse_placement(input_file)
    classify_compute_nodes(tiles, paths)
    draw(R, C, tiles, paths, save=output_file)