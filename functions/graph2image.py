'''
Author: Yishuo Wang
Date: 2025-11-21 15:12:16
LastEditors: Yishuo Wang
LastEditTime: 2025-11-21 15:16:15
FilePath: /paper/my_method/graph2image.py
Description: convert graph to image mask

Copyright (c) 2025 by Yishuo Wang, All Rights Reserved. 
'''
import numpy as np
from skimage.draw import line

# ------------------------
# graph -> mask (向量化)
# ------------------------
def graph2im_vectorized(G, shape):
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=np.uint8)
    rr_list, cc_list = [], []
    for node in G.nodes():
        pts = G.nodes[node].get('pts', None)
        if pts is not None and len(pts) > 0:
            pts = np.asarray(pts, dtype=int)
            rr = np.clip(pts[:, 0], 0, rows - 1)
            cc = np.clip(pts[:, 1], 0, cols - 1)
            mask[rr, cc] = 1
    for s, t in G.edges():
        vals = list(G[s][t].values())
        if not vals:
            continue
        for val in vals:
            coords = val.get('pts', None)
            if coords is None or len(coords) < 2:
                continue
            coords = np.asarray(coords, dtype=int)
            r0 = coords[:-1, 0]; c0 = coords[:-1, 1]
            r1 = coords[1:, 0]; c1 = coords[1:, 1]
            for y0, x0, y1, x1 in zip(r0, c0, r1, c1):
                rr_seg, cc_seg = line(int(y0), int(x0), int(y1), int(x1))
                rr_list.append(rr_seg); cc_list.append(cc_seg)
    if rr_list:
        rr_all = np.concatenate(rr_list); cc_all = np.concatenate(cc_list)
        rr_all = np.clip(rr_all, 0, rows - 1); cc_all = np.clip(cc_all, 0, cols - 1)
        mask[rr_all, cc_all] = 1
    return mask