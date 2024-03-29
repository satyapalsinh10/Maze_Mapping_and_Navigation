import networkx as nx
import matplotlib.pyplot as plt
import heapq
import numpy as np

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(point, maze_points):
    neighbors = []
    
    # Include only horizontal and vertical neighbors
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = (point[0] + dx, point[1] + dy)
        if neighbor in maze_points:
            neighbors.append(neighbor)
    
    return neighbors

def astar(maze_points, start, end):
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0, start))
    came_from = {}

    g_score = {point: float('inf') for point in maze_points}
    g_score[start] = 0

    f_score = {point: float('inf') for point in maze_points}
    f_score[start] = heuristic(start, end)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        closed_set.add(current)

        for neighbor in get_neighbors(current, maze_points):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                if neighbor not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No path found between start and end points.")
    return None

# Given maze intermediate points
maze_points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0), (35, 0), (36, 0), (37, 0), (38, 0), (39, 0), (40, 0), (41, 0), (42, 0), (43, 0), (44, 0), (45, 0), (46, 0), (47, 0), (48, 0), (49, 0), (50, 0), (51, 0), (52, 0), (53, 0), (54, 0), (55, 0), (56, 0), (57, 0), (58, 0), (59, 0), (60, 0), (61, 0), (62, 0), (63, 0), (64, 0), (65, 0), (66, 0), (67, 0), (68, 0), (69, 0), (70, 0), (71, 0), (72, 0), (73, 0), (74, 0), (75, 0), (76, 0), (77, 0), (78, 0), (79, 0), (80, 0), (81, 0), (82, 0), (83, 0), (84, 0), (85, 0), (86, 0), (0, 1), (27, 1), (29, 1), (30, 1), (86, 1), (0, 2), (27, 2), (29, 2), (30, 2), (86, 2), (0, 3), (27, 3), (29, 3), (30, 3), (86, 3), (0, 4), (27, 4), (29, 4), (30, 4), (86, 4), (0, 5), (27, 5), (29, 5), (30, 5), (86, 5), (0, 6), (27, 6), (29, 6), (30, 6), (86, 6), (0, 7), (27, 7), (29, 7), (30, 7), (86, 7), (0, 8), (27, 8), (29, 8), (30, 8), (86, 8), (0, 9), (27, 9), (29, 9), (30, 9), (86, 9), (0, 10), (27, 10), (29, 10), (30, 10), (86, 10), (0, 11), (27, 11), (28, 11), (29, 11), (30, 11), (31, 11), (32, 11), (33, 11), (34, 11), (35, 11), (36, 11), (37, 11), (38, 11), (39, 11), (40, 11), (41, 11), (86, 11), (0, 12), (29, 12), (30, 12), (41, 12), (86, 12), (0, 13), (29, 13), (30, 13), (41, 13), (86, 13), (0, 14), (29, 14), (30, 14), (41, 14), (86, 14), (0, 15), (29, 15), (30, 15), (31, 15), (32, 15), (33, 15), (34, 15), (35, 15), (36, 15), (37, 15), (38, 15), (39, 15), (40, 15), (41, 15), (42, 15), (43, 15), (86, 15), (0, 16), (30, 16), (41, 16), (43, 16), (86, 16), (0, 17), (30, 17), (41, 17), (43, 17), (86, 17), (0, 18), (30, 18), (41, 18), (43, 18), (86, 18), (30, 19), (31, 19), (32, 19), (33, 19), (34, 19), (35, 19), (36, 19), (37, 19), (38, 19), (39, 19), (40, 19), (41, 19), (42, 19), (43, 19), (44, 19), (45, 19), (46, 19), (47, 19), (48, 19), (49, 19), (50, 19), (51, 19), (52, 19), (53, 19), (54, 19), (55, 19), (56, 19), (57, 19), (58, 19), (59, 19), (86, 19), (41, 20), (43, 20), (59, 20), (86, 20), (41, 21), (43, 21), (59, 21), (86, 21), (41, 22), (43, 22), (59, 22), (86, 22), (41, 23), (43, 23), (59, 23), (86, 23), (41, 24), (43, 24), (59, 24), (86, 24), (41, 25), (43, 25), (59, 25), (86, 25), (41, 26), (43, 26), (59, 26), (86, 26), (41, 27), (43, 27), (59, 27), (86, 27), (41, 28), (43, 28), (59, 28), (86, 28), (41, 29), (43, 29), (59, 29), (86, 29), (41, 30), (43, 30), (59, 30), (86, 30), (41, 31), (43, 31), (59, 31), (86, 31), (41, 32), (43, 32), (59, 32), (60, 32), (61, 32), (62, 32), (63, 32), (64, 32), (65, 32), (66, 32), (67, 32), (68, 32), (69, 32), (70, 32), (71, 32), (72, 32), (73, 32), (86, 32), (41, 33), (43, 33), (73, 33), (86, 33), (41, 34), (43, 34), (73, 34), (86, 34), (41, 35), (43, 35), (73, 35), (86, 35), (41, 36), (43, 36), (73, 36), (86, 36), (41, 37), (43, 37), (73, 37), (86, 37), (41, 38), (43, 38), (73, 38), (86, 38), (41, 39), (43, 39), (73, 39), (86, 39), (41, 40), (43, 40), (73, 40), (86, 40), (41, 41), (43, 41), (73, 41), (86, 41), (41, 42), (43, 42), (73, 42), (86, 42), (41, 43), (42, 43), (43, 43), (44, 43), (45, 43), (46, 43), (47, 43), (48, 43), (49, 43), (50, 43), (51, 43), (52, 43), (53, 43), (54, 43), (55, 43), (56, 43), (57, 43), (58, 43), (59, 43), (60, 43), (61, 43), (62, 43), (63, 43), (64, 43), (65, 43), (66, 43), (67, 43), (68, 43), (69, 43), (70, 43), (71, 43), (72, 43), (73, 43), (74, 43), (75, 43), (76, 43), (77, 43), (78, 43), (79, 43), (80, 43), (81, 43), (82, 43), (83, 43), (84, 43), (86, 43), (73, 44), (86, 44), (54, 45), (55, 45), (56, 45), (57, 45), (58, 45), (59, 45), (60, 45), (61, 45), (62, 45), (63, 45), (64, 45), (65, 45), (66, 45), (67, 45), (68, 45), (69, 45), (70, 45), (71, 45), (72, 45), (73, 45), (86, 45), (54, 46), (86, 46), (54, 47), (86, 47), (54, 48), (86, 48), (54, 49), (86, 49), (54, 50), (86, 50), (54, 51), (86, 51), (54, 52), (86, 52), (54, 53), (86, 53), (54, 54), (86, 54), (54, 55), (86, 55), (54, 56), (86, 56), (54, 57), (86, 57), (54, 58), (86, 58), (54, 59), (86, 59), (54, 60), (86, 60), (54, 61), (55, 61), (56, 61), (57, 61), (58, 61), (59, 61), (60, 61), (61, 61), (62, 61), (63, 61), (64, 61), (65, 61), (66, 61), (67, 61), (68, 61), (69, 61), (70, 61), (71, 61), (72, 61), (73, 61), (74, 61), (75, 61), (76, 61), (77, 61), (78, 61), (79, 61), (80, 61), (81, 61), (82, 61), (83, 61), (84, 61), (85, 61), (86, 61)]

# Start and end points
start_point = (0, 0)
end_point = ((75, 61))

# Find the shortest path
shortest_path = astar(maze_points, start_point, end_point)

# Print the result
if shortest_path:
    print(f"Shortest Path: {shortest_path}")
maze_points = np.array(maze_points)
shortest_path = np.array(shortest_path)
plt.scatter(maze_points[:,1],maze_points[:,0])

# Plot shortest path with increased thickness (linewidth=2 in this case)
plt.plot( shortest_path[:, 1], shortest_path[:, 0], "r", linewidth=2)


# Show the plot
plt.show()