'''
Author: Yishuo Wang
Date: 2025-12-02 10:35:51
LastEditors: Yishuo Wang
LastEditTime: 2026-03-30 10:55:49
FilePath: /paper_detection/dataset/Cayula/cayula_plus.py
Description: the front detection method improved from cayula-1992

Copyright (c) 2025 by Yishuo Wang, All Rights Reserved. 
'''
import numpy as np 
from scipy.spatial import cKDTree

epsilon = 1e-12

def idw_interpolate(grid, k=8, power=2):
    H, W = grid.shape
    yy, xx = np.indices((H, W))

    # 已知点
    mask = ~np.isnan(grid)
    pts = np.column_stack((yy[mask], xx[mask]))
    vals = grid[mask]

    tree = cKDTree(pts)

    # 所有点
    all_pts = np.column_stack((yy.ravel(), xx.ravel()))

    dist, idx = tree.query(all_pts, k=k)
    weights = 1.0 / (dist ** power + epsilon)
    interp_vals = np.sum(weights * vals[idx], axis=1) / np.sum(weights, axis=1)

    return interp_vals.reshape(H, W)

def sied2_prob(sst, row, col, winSize=20, numTry=64,
               numClass1_threshold=0.3, numClass2_threshold=0.3):

    H, W = sst.shape

    prob = np.zeros((winSize, winSize), dtype=float)

    tl_y, tl_x = row, col
    br_y, br_x = row + winSize, col + winSize

    if br_y > H or br_x > W:
        return prob
    
    win = sst[tl_y:br_y, tl_x:br_x]
    data = win.flatten()

    if np.all(np.isnan(data)):
        return prob
    
    winMin = np.nanmin(data)
    winMax = np.nanmax(data)
    if not np.isfinite(winMin) or winMin == winMax:
        return prob

    pst = np.linspace(winMin, winMax, numTry)

    d = data[:, None]
    c1 = d < pst
    c2 = ~c1

    total_pix = winSize * winSize
    th1 = total_pix * numClass1_threshold
    th2 = total_pix * numClass2_threshold

    numC1 = np.nansum(c1, axis=0)
    numC2 = np.nansum(c2, axis=0)

    valid = (numC1 >= th1) & (numC2 >= th2)
    if not np.any(valid):
        return prob

    pst_valid = pst[valid]
    c1_valid = c1[:, valid]
    c2_valid = c2[:, valid]

    mean1 = np.nanmean(np.where(c1_valid, d, np.nan), axis=0)
    mean2 = np.nanmean(np.where(c2_valid, d, np.nan), axis=0)
    var1 = np.nanvar(np.where(c1_valid, d, np.nan), axis=0)
    var2 = np.nanvar(np.where(c2_valid, d, np.nan), axis=0)

    total = numC1[valid] + numC2[valid]
    wcv = numC1[valid] / total * var1 + numC2[valid] / total * var2
    bcv = numC1[valid] * numC2[valid] / total * (mean1 - mean2) ** 2

    theta = bcv / (wcv + bcv + epsilon)

    # 关键：对所有 pst 累加贡献
    for i in range(len(pst_valid)):
        pst_i = pst_valid[i]
        theta_i = theta[i]

        center = win >= pst_i

        up    = np.roll(win,  1, axis=0) < pst_i
        down  = np.roll(win, -1, axis=0) < pst_i
        left  = np.roll(win,  1, axis=1) < pst_i
        right = np.roll(win, -1, axis=1) < pst_i

        up[0, :] = False
        down[-1, :] = False
        left[:, 0] = False
        right[:, -1] = False

        edge = center & (up | down | left | right)

        prob += theta_i * edge

    # 归一化到 0-1
    if np.max(prob) > 0:
        prob /= np.max(prob)

    return prob

def sied2_multiscale(sst, winSize=20):
    H, W = sst.shape

    out = np.zeros((H, W), dtype=float)

    original_mask = np.isnan(sst)
    sst_filled = idw_interpolate(sst)

    for row in range(0, H, winSize):
        for col in range(0, W, winSize):
            edge20 = sied2_prob(sst_filled, row, col)
            out[row:row+winSize, col:col+winSize] = edge20

    out[original_mask] = np.nan

    return out