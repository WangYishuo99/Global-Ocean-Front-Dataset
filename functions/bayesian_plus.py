'''
Author: Yishuo Wang
Date: 2025-11-18 15:49:08
LastEditors: Yishuo Wang
LastEditTime: 2026-03-26 13:58:14
FilePath: /paper_detection/dataset/GBM/bayesian_plus.py
Description: accelerated bayesian method with vectorized LDE and BD calculation

Copyright (c) 2025 by Yishuo Wang, All Rights Reserved. 
'''
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit, prange

epsilon = np.float32(1e-6)
LDE_threshold = np.float32(0.1)
BD_threshold  = np.float32(0.1)
FOUR_OVER_SEVEN = np.float32(4.0 / 7.0)
HALF = np.float32(0.5)
ZERO = np.float32(0.0)
ONE = np.float32(1.0)

def compute_LDE_BD(SST):
    SST = np.asarray(SST, dtype=np.float32)
    rows, cols = SST.shape
    SST_pad = np.pad(SST, 1, constant_values=np.float32(np.nan))
    windows = sliding_window_view(SST_pad, (3,3))
    windows_flat = windows.reshape(rows, cols, 9)
    neighbor_vector = np.delete(windows_flat, 4, axis=2)  # 去掉中心点

    all_nan_mask = np.all(np.isnan(neighbor_vector), axis=2)

    v_max  = np.nanmax(neighbor_vector, axis=2)
    v_min  = np.nanmin(neighbor_vector, axis=2)
    v_mean = np.nanmean(neighbor_vector, axis=2)

    pair_idx = [(0,7),(1,6),(2,5),(3,4)]
    sum_lde = np.zeros((rows, cols), dtype=np.float32)
    sum_bd  = np.zeros((rows, cols), dtype=np.float32)
    count_pairs = np.zeros((rows, cols), dtype=np.float32)
    denom = v_max - v_min + epsilon

    for a,b in pair_idx:
        valid_mask = (~np.isnan(neighbor_vector[:,:,a])) & (~np.isnan(neighbor_vector[:,:,b]))
        count_pairs += valid_mask.astype(np.float32)
        diff = np.abs(neighbor_vector[:,:,a] - neighbor_vector[:,:,b])
        sum_lde += np.where(valid_mask,
                             FOUR_OVER_SEVEN * (v_max - v_mean - diff) / denom + HALF,
                             ZERO)
        sum_bd += np.where(valid_mask,
                           diff / denom,
                           ZERO)

    count_safe = count_pairs.copy()
    count_safe[count_safe == ZERO] = np.float32(np.nan)

    LDE = (sum_lde / count_safe).astype(np.float32)
    BD  = (sum_bd  / count_safe).astype(np.float32)
    LDE[all_nan_mask] = np.nan
    BD[all_nan_mask] = np.nan
    return LDE, BD

@njit(parallel=True)
def compute_posterior_batch(idx_flat, marked_matrix_out, prior_flat, LDE_flat, BD_flat, grad_flat):
    for idx_ptr in prange(len(idx_flat)):
        idx = idx_flat[idx_ptr]

        grad_p = grad_flat[idx]
        LDE_p = LDE_flat[idx]
        BD_p  = BD_flat[idx]
        prior_p = prior_flat[idx]

        # Masks for all points
        front_mask = grad_flat > grad_p
        nonfront_mask = grad_flat < grad_p

        # LDE / BD similarity masks
        delta_LDE = np.abs(LDE_flat - LDE_p) <= LDE_threshold
        delta_BD  = np.abs(BD_flat  - BD_p) <= BD_threshold

        F_count = front_mask.sum()
        NF_count = nonfront_mask.sum()

        P_front = np.float32(0.0)
        P_nonfront = np.float32(0.0)

        if F_count > 0:
            F_LDE_count = np.sum(front_mask & delta_LDE)
            F_BD_count  = np.sum(front_mask & delta_BD)
            F_count_f = np.float32(F_count)
            P_front = (np.float32(F_LDE_count) / F_count_f) * (np.float32(F_BD_count) / F_count_f)

        if NF_count > 0:
            NF_LDE_count = np.sum(nonfront_mask & delta_LDE)
            NF_BD_count  = np.sum(nonfront_mask & delta_BD)
            NF_count_f = np.float32(NF_count)
            P_nonfront = (np.float32(NF_LDE_count) / NF_count_f) * (np.float32(NF_BD_count) / NF_count_f)

        # Posterior decision
        probability_front = prior_p * P_front / (prior_p * P_front + (ONE - prior_p) * P_nonfront + epsilon)
        marked_matrix_out[idx] = probability_front

    return marked_matrix_out

def bayesian_vectorized(marked_matrix, p10, p20, gradient_magnitude, SST):
    marked_matrix = np.asarray(marked_matrix, dtype=np.float32)
    gradient_magnitude = np.asarray(gradient_magnitude, dtype=np.float32)
    SST = np.asarray(SST, dtype=np.float32)
    p10 = np.float32(p10)
    p20 = np.float32(p20)

    rows, cols = marked_matrix.shape

    prior_matrix = np.ones_like(marked_matrix, dtype=np.float32)
    mask_nan = np.isnan(marked_matrix)
    mask2 = (marked_matrix == 2)
    prior_matrix[mask_nan] = np.nan
    prior_matrix[mask2] = (gradient_magnitude[mask2] - p20) / (p10 - p20)

    LDE, BD = compute_LDE_BD(SST)

    marked_matrix_out = marked_matrix.ravel().copy()
    grad_flat = gradient_magnitude.ravel().astype(np.float32, copy=False)
    LDE_flat  = LDE.ravel().astype(np.float32, copy=False)
    BD_flat   = BD.ravel().astype(np.float32, copy=False)
    prior_flat = prior_matrix.ravel().astype(np.float32, copy=False)

    idx_flat = np.flatnonzero(mask2.flatten() & ~mask_nan.flatten())

    marked_matrix_out = compute_posterior_batch(idx_flat, marked_matrix_out, prior_flat, LDE_flat, BD_flat, grad_flat)

    return marked_matrix_out.reshape(rows, cols)
