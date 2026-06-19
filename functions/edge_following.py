'''
Author: Yishuo Wang
Date: 2024-03-12 16:39:59
LastEditors: Yishuo Wang
LastEditTime: 2024-09-05 09:34:39
FilePath: /传统方法识别/edge_following.py
Description: an edge following algorithm to merge the edges of the SST fronts, using deep first search to find the connected components

Copyright (c) 2024 by Yishuo Wang, All Rights Reserved. 
'''

# Replace recursive DFS with an iterative walk to avoid deep recursion
def walk_path(x, y, visited, current_list, marked_matrix, direction):
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    cx, cy = x, y
    while True:
        visited[cx][cy] = True
        current_list.append((cx, cy))

        pointers = []
        direction_difference = []
        for i in range(8):
            nx, ny = cx + dx[i], cy + dy[i]
            if (
                0 <= nx < len(marked_matrix)
                and 0 <= ny < len(marked_matrix[0])
                and marked_matrix[nx][ny] == 1
                and not visited[nx][ny]
            ):
                pointers.append(i)
                direction_difference.append(abs(direction[cx][cy] - direction[nx][ny]))

        if not pointers:
            return current_list

        # Choose neighbour with minimum direction difference
        min_index = direction_difference.index(min(direction_difference))
        cx, cy = cx + dx[pointers[min_index]], cy + dy[pointers[min_index]]

# find the fronts and save them in a list, the list is nested, each element is a list of points of a front
def follow(marked_matrix, direction):
    # Initialize the visited matrix
    visited = [[False]*len(marked_matrix[0]) for _ in range(len(marked_matrix))]
    
    nested_list = []
    
    # Scan the marked_matrix
    for i in range(len(marked_matrix)):
        for j in range(len(marked_matrix[0])):
            # If the point is 1 and has not been visited, start a DFS from it
            if marked_matrix[i][j] == 1 and not visited[i][j]:
                current_list = []
                current_list = walk_path(i, j, visited, current_list, marked_matrix, direction)
                nested_list.append(current_list)
    
    return nested_list