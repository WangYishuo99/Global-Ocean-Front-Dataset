#
# @Author: Yishuo Wang
# @Date: 2026-06-19 13:54:20
# @LastEditors: Yishuo Wang
# @LastEditTime: 2026-06-19 16:27:57
# @Description: GBM and Canny edge detection algorithm for front detection
# @Company: Shanghai Jiao Tong University
#
import numpy as np
import functions.gradient_calculation as gradient_calculation
import functions.threshold as threshold
import functions.bayesian_plus as bayesian_plus

# Define the thresholds
upper_threshold = np.float32(90.0)
lower_threshold = np.float32(80.0)

def get_gradient_front_zone(data, lon, lat, beginLon, endLon, beginLat, endLat):
    # 1. prepossess and filter the data
    lon_index = np.where((lon >= beginLon) & (lon < endLon))[0]
    lat_index = np.where((lat >= beginLat) & (lat < endLat))[0]

    lon_extracted = np.float32(np.asarray(lon[lon_index]).reshape(-1))
    lat_extracted = np.float32(np.asarray(lat[lat_index]).reshape(-1))

    data_temp = data.variables['analysed_sst'][:, lat_index, lon_index]
    data_temp = data_temp[0]
    data_temp = np.array(data_temp, dtype=np.float32)
    data_temp[np.where(data_temp > 100)] = np.nan
    data_temp[np.where(data_temp < -100)] = np.nan

    # 2. gradient
    gradient_magnitude, gradient_direction = gradient_calculation.magnitude_and_direction_gradient(data_temp)
    
    # 3. front_zone
    marked_matrix, p10, p20 = threshold.thresholds_calculation(gradient_magnitude, upper_threshold, lower_threshold)

    gbm_matrix = bayesian_plus.bayesian_vectorized(marked_matrix, p10, p20, gradient_magnitude, data_temp)

    gbm_matrix[np.where(gbm_matrix == 2)] = 0
    
    return gradient_magnitude, gradient_direction, gbm_matrix, lon_extracted, lat_extracted