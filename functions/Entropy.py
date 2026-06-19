'''
Author: Yishuo Wang
Date: 2025-12-04 18:11:31
LastEditors: Yishuo Wang
LastEditTime: 2026-03-31 13:10:00
FilePath: /paper_dataset/Entropy/entropy_plus.py
Description: vectorized implementation of entropy.py

Copyright (c) 2025 by Yishuo Wang, All Rights Reserved. 
'''
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

epsilon = np.float32(1e-12)
one = np.float32(1.0)
half = np.float32(0.5)
window_size = 10

def idw_interpolate(grid, k=8, power=2):
    grid = grid.astype(np.float32, copy=False)
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
    dist = dist.astype(np.float32, copy=False)
    weights = one / (dist ** power + epsilon)
    interp_vals = np.sum(weights * vals[idx], axis=1, dtype=np.float32) / np.sum(weights, axis=1, dtype=np.float32)
    
    return interp_vals.reshape(H, W).astype(np.float32, copy=False)

def jensen_shannon_vec(P, Q):
    # P, Q: (..., N)
    P = np.clip(P, epsilon, one)
    Q = np.clip(Q, epsilon, one)
    M = half * (P + Q)
    js = half * np.sum(P * np.log(P / M), axis=-1, dtype=np.float32) + \
        half * np.sum(Q * np.log(Q / M), axis=-1, dtype=np.float32)
    return js

def normalize_window(win):
    # win: (..., window_size, window_size)
    win = np.nan_to_num(win, nan=0.0, copy=False)
    win = win.astype(np.float32, copy=False)

    win = win - np.min(win, axis=(-2, -1), keepdims=True)

    s = np.sum(win, axis=(-2, -1), keepdims=True, dtype=np.float32)
    s = np.where(s == 0, one, s)
    
    return win / s

def get_front_zone(raw_data):
    raw_data = raw_data.astype(np.float32, copy=False)
    rows, cols = raw_data.shape
    half_w = window_size // 2

    original_mask = np.isnan(raw_data)
    raw_data = idw_interpolate(raw_data)

    # 左右窗口
    pad_lr = ((half_w, half_w), (half_w*2, half_w*2))
    data_pad_lr = np.pad(raw_data, pad_lr, constant_values=np.nan)
    lr_windows = sliding_window_view(data_pad_lr, (window_size, window_size))
    left_win  = lr_windows[:rows, :cols, :, :]
    right_win = lr_windows[:rows, half_w*2:half_w*2+cols, :, :]

    mask_lr = ~np.isnan(left_win).any(axis=(2,3)) & ~np.isnan(right_win).any(axis=(2,3))

    left_prob  = normalize_window(np.where(mask_lr[..., None, None], left_win, 0))
    right_prob = normalize_window(np.where(mask_lr[..., None, None], right_win, 0))

    left_flat  = left_prob.reshape(rows, cols, -1)
    right_flat = right_prob.reshape(rows, cols, -1)

    js_lr = np.full((rows, cols), np.nan, dtype=np.float32)
    valid_lr = mask_lr
    js_lr[valid_lr] = jensen_shannon_vec(left_flat[valid_lr], right_flat[valid_lr])

    # 上下窗口
    pad_ul = ((half_w*2, half_w*2), (half_w, half_w))
    data_pad_ul = np.pad(raw_data, pad_ul, constant_values=np.nan)
    ul_windows = sliding_window_view(data_pad_ul, (window_size, window_size))
    upper_win = ul_windows[:rows, :cols, :, :]
    lower_win = ul_windows[half_w*2:half_w*2+rows, :cols, :, :]

    mask_ul = ~np.isnan(upper_win).any(axis=(2,3)) & ~np.isnan(lower_win).any(axis=(2,3))
    upper_prob = normalize_window(np.where(mask_ul[..., None, None], upper_win, 0))
    lower_prob = normalize_window(np.where(mask_ul[..., None, None], lower_win, 0))

    upper_flat = upper_prob.reshape(rows, cols, -1)
    lower_flat = lower_prob.reshape(rows, cols, -1)

    js_ul = np.full((rows, cols), np.nan, dtype=np.float32)
    valid_ul = mask_ul
    js_ul[valid_ul] = jensen_shannon_vec(upper_flat[valid_ul], lower_flat[valid_ul])

    # 最大 JS 散度
    js_total = np.nanmax(np.stack([js_lr, js_ul]), axis=0)
    js_total[original_mask] = np.nan
    
    return js_total.astype(np.float32, copy=False)