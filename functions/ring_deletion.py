'''
Author: Yishuo Wang
Date: 2024-11-29 12:31:38
LastEditors: Yishuo Wang
LastEditTime: 2024-11-29 17:21:57
FilePath: /blue_sea_sky_v16/ring_deletion.py
Description: the function is to delete the ring structures in the fronts

Copyright (c) 2024 by Yishuo Wang, All Rights Reserved. 
'''

def delete_ring(fronts):

    for i in range(len(fronts)):
        current_front = fronts[i]
        # if the front is shorter than 4, it is not a ring
        if len(current_front) < 4:
            continue

        # 1. if the start and end are neighbour, it is a ring
        start_x, start_y = current_front[0]
        end_x, end_y = current_front[-1]
        delta_x = abs(start_x - end_x)
        delta_y = abs(start_y - end_y)

        if (delta_x == 1 and delta_y == 0) or (delta_x == 0 and delta_y == 1) or (delta_x == 1 and delta_y == 1):
            fronts[i] = []
            continue
        
        # 2. discard partial ring
        for j in range(1, len(current_front)-2):
            x = current_front[j][0]
            y = current_front[j][1]
            delta_x = abs(x - end_x)
            delta_y = abs(y - end_y)
            if (delta_x == 1 and delta_y == 0) or (delta_x == 0 and delta_y == 1) or (delta_x == 1 and delta_y == 1):
                # delete the ring(current_front[j:-1])
                fronts[i] = current_front[:j]

        # 3. flip the current_front and discard partial ring
        current_front = current_front[::-1]
        end_x, end_y = current_front[-1]

        for j in range(1, len(current_front)-2):
            x = current_front[j][0]
            y = current_front[j][1]
            delta_x = abs(x - end_x)
            delta_y = abs(y - end_y)
            if (delta_x == 1 and delta_y == 0) or (delta_x == 0 and delta_y == 1) or (delta_x == 1 and delta_y == 1):
                # delete the ring(current_front[j:-1])
                fronts[i] = current_front[:j]
                
    # delete the empty front
    fronts = [front for front in fronts if len(front) > 0]
    return fronts
