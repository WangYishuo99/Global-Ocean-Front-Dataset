#
# @Author: Yishuo Wang
# @Date: 2026-05-15 13:28:08
# @LastEditors: Yishuo Wang
# @LastEditTime: 2026-05-15 13:28:19
# @Description: quantile normalization for the frontal features
# @Company: Shanghai Jiao Tong University
#
import numpy as np

q_low = 0.05
q_high = 0.95
epislon = 1e-12

def quantile_normalize(x):
    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + epislon)