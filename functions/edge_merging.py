'''
Author: Yishuo Wang
Date: 2024-04-17 15:00:32
LastEditors: Yishuo Wang
LastEditTime: 2025-11-25 18:45:26
FilePath: /paper/my_method/edge_merging.py
Description: an edge merging algorithm to merge the edges of the SST fronts

Copyright (c) 2024 by Yishuo Wang, All Rights Reserved. 
'''
import functions.edge_filling as edge_filling

def merge(fronts, merge_radius, direction):
    # set a flag indicating if the front has changed, if not changed, the loop will stop, and the mergeing procedure is done
    flag = True

    # change the fronts overtime
    while flag:
        flag = False
        
        for i in range(len(fronts)):
            # get the current front
            current_front = fronts[i]

            # set a list to store the indexes of the fronts that need to be merged on standby
            merge_list = []

            # set a mode list, 1 rep head to head, 2 rep head to tail, 3 rep tail to head, 4 rep tail to tail
            mode = []
            
            # 1. get the current front's head
            current_front_first = current_front[0]
            x_first, y_first = current_front_first

            for j in range(i+1, len(fronts)):
                # get the next front
                next_front = fronts[j]

                # 1.1 get the next front's head
                next_front_first = next_front[0]
                x_next_head, y_next_head = next_front_first
                if abs(x_first - x_next_head) <= merge_radius and abs(y_first - y_next_head) <= merge_radius:
                    merge_list.append(j)
                    mode.append(1)
                
                # 1.2 get the next front's tail
                # if it is not 1 pixel, the tail should also be considered
                if len(next_front) > 1:
                    next_front_last = next_front[-1]
                    x_next_tail, y_next_tail = next_front_last
                    if abs(x_first - x_next_tail) <= merge_radius and abs(y_first - y_next_tail) <= merge_radius:
                        merge_list.append(j)
                        mode.append(2)

            # 2. get the current front's tail
            if len(current_front) > 1:
                current_front_last = current_front[-1]
                x_last, y_last = current_front_last

                for j in range(i+1, len(fronts)):
                    # get the next front
                    next_front = fronts[j]

                    # 2.1 get the next front's head
                    next_front_first = next_front[0]
                    x_next_head, y_next_head = next_front_first
                    if abs(x_last - x_next_head) <= merge_radius and abs(y_last - y_next_head) <= merge_radius:
                        merge_list.append(j)
                        mode.append(3)

                    # 2.2 get the next front's tail
                    # if it is not 1 pixel, the tail should also be considered
                    if len(next_front) > 1:
                        next_front_last = next_front[-1]
                        x_next_tail, y_next_tail = next_front_last
                        if abs(x_last - x_next_tail) <= merge_radius and abs(y_last - y_next_tail) <= merge_radius:
                            merge_list.append(j)
                            mode.append(4)
            
            # no candiate fronts to be merged
            if not merge_list:
                continue

            # 3. then traverse the merge_list and mode list to select an endpoint to merge
            # set a distance list the size of the merge_list
            distance = [0]*len(merge_list)
            # set a direction list
            direction_list = [0]*len(merge_list)

            # calculate the distance and direction
            for j in range(len(merge_list)):
                # get the next front
                next_front = fronts[merge_list[j]]

                # get the mode
                current_mode = mode[j]

                # 1. head to head
                if current_mode == 1:
                    # calculate the distance
                    distance[j] = (current_front[0][0] - next_front[0][0])**2 + (current_front[0][1] - next_front[0][1])**2
                    # calculate the direction
                    direction_list[j] = abs(direction[current_front[0][0]][current_front[0][1]] - direction[next_front[0][0]][next_front[0][1]])
                # 2. head to tail
                elif current_mode == 2:
                    # calculate the distance
                    distance[j] = (current_front[0][0] - next_front[-1][0])**2 + (current_front[0][1] - next_front[-1][1])**2
                    # calculate the direction
                    direction_list[j] = abs(direction[current_front[0][0]][current_front[0][1]] - direction[next_front[-1][0]][next_front[-1][1]])
                # 3. tail to head
                elif current_mode == 3:
                    # calculate the distance
                    distance[j] = (current_front[-1][0] - next_front[0][0])**2 + (current_front[-1][1] - next_front[0][1])**2
                    # calculate the direction
                    direction_list[j] = abs(direction[current_front[-1][0]][current_front[-1][1]] - direction[next_front[0][0]][next_front[0][1]])
                # 4. tail to tail
                elif current_mode == 4:
                    # calculate the distance
                    distance[j] = (current_front[-1][0] - next_front[-1][0])**2 + (current_front[-1][1] - next_front[-1][1])**2
                    # calculate the direction
                    direction_list[j] = abs(direction[current_front[-1][0]][current_front[-1][1]] - direction[next_front[-1][0]][next_front[-1][1]])

            # then comes the judgemnet 1. minimum distance 2. minimum direction
            # sort them by distance and direction
            merge_list_combined = [(merge_list[i], distance[i], direction_list[i]) for i in range(len(merge_list))]
            merge_list_combined.sort(key=lambda x: (x[1], x[2]))
            merge_list = [p[0] for p in merge_list_combined]
            mode_list_combined = [(mode[i], distance[i], direction_list[i]) for i in range(len(mode))]
            mode_list_combined.sort(key=lambda x: (x[1], x[2]))
            mode = [p[0] for p in mode_list_combined]
            
            # 4. finally, merge the two fronts
            for j in range(len(merge_list)):
                index_found = merge_list[j]
                current_mode = mode[j]
                # if it can not merge, then continue to the next front on standby
                if edge_filling.fill(i, index_found, fronts, current_mode, direction):
                    continue
                # if it can merge, then break the loop
                else:
                    # flag_at_least_once = True
                    # j = -1证明已经合并成功，跳出循环
                    j = -1
                    break
            
            # if it changed, then the loop should restart
            if j == -1:
                flag = True
                break
            
    return