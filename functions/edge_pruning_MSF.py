'''
Author: Yishuo Wang
Date: 2025-11-21 14:56:47
LastEditors: Yishuo Wang
LastEditTime: 2025-12-20 15:14:08
FilePath: /paper_detection/flowchart/edge_pruning_MSF.py
Description: edge pruning based on Maximum Spanning Forest (MSF)

Copyright (c) 2025 by Yishuo Wang, All Rights Reserved. 
'''
import numpy as np
import networkx as nx
import sknw
import functions.graph2image as graph2image

# ------------------------------------------------------------
# Branch weight: thickness^alpha × length^beta
# ------------------------------------------------------------
def compute_branch_weight(pts, dist_map, alpha=1.0, beta=1.0):
    """
    pts: (N, 2) array of skeleton pixels (row, col)
    dist_map: distance transform (local thickness)
    """
    pts = np.asarray(pts, dtype=int)
    r, c = pts[:, 0], pts[:, 1]

    length = len(pts)
    thickness = np.mean(dist_map[r, c])

    return (thickness ** alpha) * (length ** beta)


# ------------------------------------------------------------
# MSF-based spider-web pruning + 1-D backbone extraction
# ------------------------------------------------------------
def prune_spider_web_by_msf(
    skel_mask,
    dist_map,
    alpha=1.0,
    beta=1.0,
    debug=False
):
    """
    Parameters
    ----------
    skel_mask : binary skeleton mask
    dist_map  : distance transform (same shape)
    alpha, beta : weight exponents
    """

    # =========================================================
    # Step 0: Skeleton graph (multigraph)
    # =========================================================
    Gm = sknw.build_sknw(skel_mask, multi=True)

    # =========================================================
    # Step 1: Multigraph → weighted simple graph
    #         (keep the strongest parallel edge)
    # =========================================================
    G = nx.Graph()

    for u, v, k, data in Gm.edges(keys=True, data=True):
        pts = data['pts']
        w = compute_branch_weight(pts, dist_map, alpha, beta)

        if G.has_edge(u, v):
            if w > G[u][v]['weight']:
                G[u][v].update(weight=w, pts=pts)
        else:
            G.add_edge(u, v, weight=w, pts=pts)

    if debug:
        print(f"[MSF] nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # =========================================================
    # Step 2: Maximum Spanning Forest (non-iterative)
    # =========================================================
    MSF = nx.maximum_spanning_tree(G, weight='weight')

    # =========================================================
    # Step 3: Per-component backbone extraction (tree diameter)
    # =========================================================
    backbone_edges = set()

    for comp in nx.connected_components(MSF):
        if len(comp) <= 1:
            continue

        subG = MSF.subgraph(comp)

        # ---- First BFS
        start = next(iter(subG.nodes))
        dist1 = nx.single_source_shortest_path_length(subG, start)
        u = max(dist1, key=dist1.get)

        # ---- Second BFS
        paths = nx.single_source_shortest_path(subG, u)
        v = max(paths, key=lambda x: len(paths[x]))

        backbone_path = paths[v]

        # ---- Record backbone edges
        for i in range(len(backbone_path) - 1):
            a = backbone_path[i]
            b = backbone_path[i + 1]
            backbone_edges.add((min(a, b), max(a, b)))

    if debug:
        print(f"[MSF] backbone edges kept = {len(backbone_edges)}")

    # =========================================================
    # Step 4: Build pruned multigraph (only backbone)
    # =========================================================
    pruned = nx.MultiGraph()

    for u, v, k, data in Gm.edges(keys=True, data=True):
        key = (min(u, v), max(u, v))
        if key in backbone_edges:
            pruned.add_edge(u, v, pts=data['pts'])

    # =========================================================
    # Step 5: Graph → binary mask
    # =========================================================
    final_mask = graph2image.graph2im_vectorized(
        pruned, skel_mask.shape
    )

    return final_mask, pruned
