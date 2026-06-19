#
# @Author: Yishuo Wang
# @Date: 2026-04-24 13:53:01
# @LastEditors: Yishuo Wang
# @LastEditTime: 2026-06-19 14:01:19
# @Description: gradient calculation function, using Sobel operator to calculate the gradient magnitude and direction
# @Company: Shanghai Jiao Tong University
#
from scipy.ndimage import sobel
import numpy as np

resolution = np.float32(5.0)

def magnitude_and_direction_gradient(SST):
    sst32 = np.asarray(SST, dtype=np.float32)

    gx = sobel(sst32, axis=0, output=np.float32)
    gy = sobel(sst32, axis=1, output=np.float32)

    gm = np.empty_like(sst32, dtype=np.float32)
    np.hypot(gx, gy, out=gm)          
    gm /= resolution                  

    gd = np.empty_like(sst32, dtype=np.float32)
    np.arctan2(gy, gx, out=gd)        

    return gm, gd