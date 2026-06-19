'''
Author: Yishuo Wang
Date: 2024-04-17 20:55:41
LastEditors: Yishuo Wang
LastEditTime: 2024-11-29 15:45:33
FilePath: /blue_sea_sky_v16/edge_filling.py
Description: an edge filling algorithm to fill the gaps between the SST fronts, link the ends of the SST fronts

Copyright (c) 2024 by Yishuo Wang, All Rights Reserved. 
'''
import numpy as np

# prevent it from connecting to other fronts
def judge_filled2others(x, y, fronts, direction):
    # type list will change the value of the original list, so we can not use the remove method for fronts
    # then judge if (x, y) is in the fronts or 
    for j in range(len(fronts)):
        # if (x, y) is in the other fronts or connected to nan, return True
        if (x, y) in fronts[j] or np.isnan(direction[x][y]):
            return True
    return False

def fill(i, index_found, fronts, current_mode, direction):
    current_front = fronts[i]
    next_front = fronts[index_found]

    # convert all of them to tail to head(mode 3)
    # 4.1 head to head
    if current_mode == 1:
        # flip the front, cauz we need to connect the head of the current front to the head of the next front
        current_front = current_front[::-1]
    # 4.2 head to tail
    elif current_mode == 2:
        # flip the front, cauz we need to connect the head of the current front to the head of the next front
        current_front = current_front[::-1]
        next_front = next_front[::-1]
    # 4.4 tail to tail
    elif current_mode == 4:
        # flip the front, cauz we need to connect the head of the current front to the head of the next front
        next_front = next_front[::-1]
        
    # calculate the x and y difference
    current_x = current_front[-1][0]
    current_y = current_front[-1][1]
    x_next_head = next_front[0][0]
    y_next_head = next_front[0][1] 
    delta_x = x_next_head - current_x
    delta_y = y_next_head - current_y

    # 0. if there is no gap between the two fronts, no need to fill, just connect them
    if (delta_x == 0 and abs(delta_y) == 1) or (delta_y == 0 and abs(delta_x) == 1) or (abs(delta_x) == 1 and abs(delta_y) == 1):
        merged_front = current_front + next_front
        # substitute the merged front to the current front
        fronts[i] = merged_front
        # delete the next front
        fronts.pop(index_found)
        # if it has been changed, return false
        return False

    # 1. if the pixels needed to be filled is in the diagonal direction
    if abs(delta_x) == abs(delta_y):
        # the 1 quadrant
        if delta_x > 0 and delta_y > 0:
            for k in range(1, abs(delta_x)):
                if judge_filled2others(current_x + k, current_y + k, fronts, direction):
                    return True
                current_front.append((current_x + k, current_y + k))
        # the 2 quadrant
        elif delta_x < 0 and delta_y > 0:
            for k in range(1, abs(delta_x)):
                if judge_filled2others(current_x - k, current_y + k, fronts, direction):
                    return True
                current_front.append((current_x - k, current_y + k))
        # the 3 quadrant
        elif delta_x < 0 and delta_y < 0:
            for k in range(1, abs(delta_x)):
                if judge_filled2others(current_x - k, current_y - k, fronts, direction):
                    return True
                current_front.append((current_x - k, current_y - k))
        # the 4 quadrant
        elif delta_x > 0 and delta_y < 0:
            for k in range(1, abs(delta_x)):
                if judge_filled2others(current_x + k, current_y - k, fronts, direction):
                    return True
                current_front.append((current_x + k, current_y - k))
                
    # 2. if the pixels needed to be filled is in the horizontal or vertical direction
    elif delta_x == 0:
        if delta_y > 0:
            for k in range(1, delta_y):
                if judge_filled2others(current_x, current_y + k, fronts, direction):
                    return True
                current_front.append((current_x, current_y + k))
        else:
            for k in range(1, -delta_y):
                if judge_filled2others(current_x, current_y - k, fronts, direction):
                    return True
                current_front.append((current_x, current_y - k))

    elif delta_y == 0:
        if delta_x > 0:
            for k in range(1, delta_x):
                if judge_filled2others(current_x + k, current_y, fronts, direction):
                    return True
                current_front.append((current_x + k, current_y))
        else:
            for k in range(1, -delta_x):
                if judge_filled2others(current_x - k, current_y, fronts, direction):
                    return True
                current_front.append((current_x - k, current_y))

    # 3. if the pixels needed to be filled is in the lower trainangle direction
    elif abs(delta_x) > abs(delta_y):
        # the 1 quadrant
        if delta_x > 0 and delta_y > 0:
            # along the diagonal
            for k in range(1, abs(delta_y)+1):
                if judge_filled2others(current_x + k, current_y + k, fronts, direction):
                    return True
                current_front.append((current_x + k, current_y + k))
            # if the right is the next front, no need to do anything
            if k + 1 == delta_x:
                pass
            # along the horizontal
            else:
                for p in range(k + 1, abs(delta_x)):
                    if judge_filled2others(current_x + p, current_y + k, fronts, direction):
                        return True
                    current_front.append((current_x + p, current_y + k))
                    
        # the 2 quadrant
        elif delta_x < 0 and delta_y > 0:
            for k in range(1, abs(delta_y)+1):
                if judge_filled2others(current_x - k, current_y + k, fronts, direction):
                    return True
                current_front.append((current_x - k, current_y + k))
            if k + 1 == - delta_x:
                pass
            else:
                for p in range(k + 1, abs(delta_x)):
                    if judge_filled2others(current_x - p, current_y + k, fronts, direction):
                        return True
                    current_front.append((current_x - p, current_y + k))

        # the 3 quadrant
        elif delta_x < 0 and delta_y < 0:
            for k in range(1, abs(delta_y)+1):
                if judge_filled2others(current_x - k, current_y - k, fronts, direction):
                    return True
                current_front.append((current_x - k, current_y - k))
            if k + 1 == - delta_x:
                pass
            else:
                for p in range(k + 1, abs(delta_x)):
                    if judge_filled2others(current_x - p, current_y - k, fronts, direction):
                        return True
                    current_front.append((current_x - p, current_y - k))

        # the 4 quadrant
        elif delta_x > 0 and delta_y < 0:
            for k in range(1, abs(delta_y)+1):
                if judge_filled2others(current_x + k, current_y - k, fronts, direction):
                    return True
                current_front.append((current_x + k, current_y - k))
            if k + 1 == delta_x:
                pass
            else:
                for p in range(k + 1, abs(delta_x)):
                    if judge_filled2others(current_x + p, current_y - k, fronts, direction):
                        return True
                    current_front.append((current_x + p, current_y - k))
    
    # 4. if the pixels needed to be filled is in the upper trainangle direction
    elif abs(delta_x) < abs(delta_y):
        # the 1 quadrant
        if delta_x > 0 and delta_y > 0:
            # along the diagonal
            for k in range(1, abs(delta_x)+1):
                if judge_filled2others(current_x + k, current_y + k, fronts, direction):
                    return True
                current_front.append((current_x + k, current_y + k))
            # if the upper is the next front, no need to do anything
            if k + 1 == delta_y:
                pass
            # along the horizontal
            else:
                for p in range(k + 1, abs(delta_y)):
                    if judge_filled2others(current_x + k, current_y + p, fronts, direction):
                        return True
                    current_front.append((current_x + k, current_y + p))
                    
        # the 2 quadrant
        elif delta_x < 0 and delta_y > 0:
            for k in range(1, abs(delta_x)+1):
                if judge_filled2others(current_x - k, current_y + k, fronts, direction):
                    return True
                current_front.append((current_x - k, current_y + k))
            if k + 1 == delta_y:
                pass
            else:
                for p in range(k + 1, abs(delta_y)):
                    if judge_filled2others(current_x - k, current_y + p, fronts, direction):
                        return True
                    current_front.append((current_x - k, current_y + p))

        # the 3 quadrant
        elif delta_x < 0 and delta_y < 0:
            for k in range(1, abs(delta_x)+1):
                if judge_filled2others(current_x - k, current_y - k, fronts, direction):
                    return True
                current_front.append((current_x - k, current_y - k))
            if k + 1 == - delta_y:
                pass
            else:
                for p in range(k + 1, abs(delta_y)):
                    if judge_filled2others(current_x - k, current_y - p, fronts, direction):
                        return True
                    current_front.append((current_x - k, current_y - p))

        # the 4 quadrant
        elif delta_x > 0 and delta_y < 0:
            for k in range(1, abs(delta_x)+1):
                if judge_filled2others(current_x + k, current_y - k, fronts, direction):
                    return True
                current_front.append((current_x + k, current_y - k))
            if k + 1 == - delta_y:
                pass
            else:
                for p in range(k + 1, abs(delta_y)):
                    if judge_filled2others(current_x + k, current_y - p, fronts, direction):
                        return True
                    current_front.append((current_x + k, current_y - p))                

    merged_front = current_front + next_front
    # substitute the merged front to the current front
    fronts[i] = merged_front
    # delete the next front
    fronts.pop(index_found)
    # if it has been changed, return false
    return False