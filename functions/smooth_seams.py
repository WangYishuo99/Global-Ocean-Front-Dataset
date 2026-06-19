#
# @Author: Yishuo Wang
# @Date: 2026-05-28 11:46:13
# @LastEditors: Yishuo Wang
# @LastEditTime: 2026-05-28 13:00:40
# @Description: smooth seams
# @Company: Shanghai Jiao Tong University
#
import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_tile_seams_adaptive(matrix, step, blend_width, sigma, preserve_strength):
    mat = matrix.astype(np.float32, copy=True)
    nan_mask = np.isnan(mat)

    mask = (~nan_mask).astype(np.float32)
    filled = np.nan_to_num(mat, nan=0.0).astype(np.float32)
    smooth = gaussian_filter(filled, sigma=sigma, mode='constant', cval=0.0)
    norm = gaussian_filter(mask, sigma=sigma, mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.where(norm > 1e-6, smooth / norm, np.nan).astype(np.float32)

    ref_filled = np.nan_to_num(ref, nan=0.0)
    gy, gx = np.gradient(ref_filled)
    grad = np.hypot(gy, gx)
    gmax = np.nanpercentile(grad, 99.0)
    gnorm = np.clip(grad / (gmax + 1e-6), 0.0, 1.0)

    edge_factor = 1.0 - preserve_strength * gnorm

    out = mat.copy()
    height, width = mat.shape

    for seam in range(step, width, step):
        left = max(seam - blend_width, 0)
        right = min(seam + blend_width, width)
        if left >= seam or right <= seam:
            continue

        cols = np.arange(left, right)
        dist = np.abs(cols - seam)
        alpha_cols = np.zeros_like(dist, dtype=np.float32)
        mask_in = dist <= blend_width
        alpha_cols[mask_in] = 0.5 * (1 + np.cos(np.pi * dist[mask_in] / (blend_width)))

        alpha_map = (alpha_cols[None, :] * edge_factor[:, left:right]).astype(np.float32)

        orig = out[:, left:right]
        ref_seg = ref[:, left:right]
        merged = np.where(np.isnan(orig), ref_seg, orig * (1.0 - alpha_map) + ref_seg * alpha_map)
        out[:, left:right] = merged

    for seam in range(step, height, step):
        top = max(seam - blend_width, 0)
        bottom = min(seam + blend_width, height)
        if top >= seam or bottom <= seam:
            continue

        rows = np.arange(top, bottom)
        dist = np.abs(rows - seam)
        alpha_rows = np.zeros_like(dist, dtype=np.float32)
        mask_in = dist <= blend_width
        alpha_rows[mask_in] = 0.5 * (1 + np.cos(np.pi * dist[mask_in] / (blend_width)))

        alpha_map = (alpha_rows[:, None] * edge_factor[top:bottom, :]).astype(np.float32)

        orig = out[top:bottom, :]
        ref_seg = ref[top:bottom, :]
        merged = np.where(np.isnan(orig), ref_seg, orig * (1.0 - alpha_map) + ref_seg * alpha_map)
        out[top:bottom, :] = merged

    out[nan_mask] = np.nan
    out = np.clip(out, 0.0, 1.0)
    return out

def seam_energy(mat, step):
    energies = []

    h, w = mat.shape

    # vertical seams
    for s in range(step, w, step):
        if s <= 0 or s >= w:
            continue

        left = mat[:, s-1]
        right = mat[:, s]

        mask = ~np.isnan(left) & ~np.isnan(right)

        diff = np.abs(left[mask] - right[mask])

        energies.append(np.mean(diff))

    # horizontal seams
    for s in range(step, h, step):
        if s <= 0 or s >= h:
            continue

        top = mat[s-1, :]
        bottom = mat[s, :]

        mask = ~np.isnan(top) & ~np.isnan(bottom)

        diff = np.abs(top[mask] - bottom[mask])

        energies.append(np.mean(diff))

    return np.mean(energies)

def seam_gradient_energy(matrix, step):
    mat = matrix.astype(np.float32)

    # global gradients
    gy, gx = np.gradient(mat)

    energies = []

    h, w = mat.shape

    # -------------------------
    # Vertical seams
    # -------------------------
    for s in range(step, w, step):

        if s <= 1 or s >= w - 1:
            continue

        # compare x-gradient across seam
        g_left = gx[:, s - 1]
        g_right = gx[:, s]

        mask = ~np.isnan(g_left) & ~np.isnan(g_right)

        if np.any(mask):
            diff = np.abs(g_left[mask] - g_right[mask])
            energies.append(np.mean(diff))

    # -------------------------
    # Horizontal seams
    # -------------------------
    for s in range(step, h, step):

        if s <= 1 or s >= h - 1:
            continue

        # compare y-gradient across seam
        g_top = gy[s - 1, :]
        g_bottom = gy[s, :]

        mask = ~np.isnan(g_top) & ~np.isnan(g_bottom)

        if np.any(mask):
            diff = np.abs(g_top[mask] - g_bottom[mask])
            energies.append(np.mean(diff))

    return np.mean(energies)