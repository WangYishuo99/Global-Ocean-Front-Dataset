#
# @Author: Yishuo Wang
# @Date: 2026-04-24 13:53:01
# @LastEditors: Yishuo Wang
# @LastEditTime: 2026-04-29 12:41:22
# @Description: threshold calculation and marking the front zone
# @Company: Shanghai Jiao Tong University
#

import numpy as np

# calculate the thresholds and mark the gradient_magnitude
def thresholds_calculation(gradient_magnitude, upper_threshold, lower_threshold):
    gradient_magnitude = np.asarray(gradient_magnitude, dtype=np.float32)

    # Calculate the upper and lower percentiles
    p10 = np.float32(np.nanpercentile(gradient_magnitude, upper_threshold))
    p20 = np.float32(np.nanpercentile(gradient_magnitude, lower_threshold))

    # Create a new matrix with the same shape as gradient_magnitude
    marked_matrix = np.zeros_like(gradient_magnitude, dtype=np.float32)

    # Mark the top 10% as 1
    marked_matrix[gradient_magnitude > p10] = np.float32(1.0)

    # Mark the 10% to 20% as 2
    marked_matrix[(gradient_magnitude > p20) & (gradient_magnitude <= p10)] = np.float32(2.0)

    # Reserve the Nan values
    marked_matrix[np.isnan(gradient_magnitude)] = np.nan

    return marked_matrix, p10, p20