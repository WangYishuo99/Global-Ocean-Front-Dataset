'''
Author: Yishuo Wang
Date: 2025-11-27 13:45:08
LastEditors: Yishuo Wang
LastEditTime: 2025-11-27 13:46:33
FilePath: /paper_detection/methods/GBM/front_matrix_construction.py
Description: from vector to front matrix

Copyright (c) 2025 by Yishuo Wang, All Rights Reserved. 
'''
import numpy as np

def construction(fronts, marked_matrix):
    all_coords = np.array([coord for sublist in fronts for coord in sublist])
    x_idx = all_coords[:, 0]
    y_idx = all_coords[:, 1]
    # front_matrix 必须是 float 类型才能存 NaN
    front_matrix = np.zeros_like(marked_matrix, dtype=float)
    front_matrix[x_idx, y_idx] = 1
    # 设置 NaN
    front_matrix[np.isnan(marked_matrix)] = np.nan
    return front_matrix