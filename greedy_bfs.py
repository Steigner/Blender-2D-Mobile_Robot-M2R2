# setup
blender = False

if blender:
    import bpy
else:
    import matplotlib.pyplot as plt

import math

class Node(object):
    def __init__(self, x, y, cost, pind, parent):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind
        self.parent = parent

class BestFirstSearchPlanner(object):
    def __init__(self, ox, oy, reso, rr):
        """
        Initialize grid map for greedy best-first planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        # dx, dy, cost
        self.motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)]
        ]

    def planning(self, sx, sy, gx, gy):
        """
        Greedy Best-First search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = Node(
            self.calc_xyindex(sx, self.minx),
            self.calc_xyindex(sy, self.miny), 
            0.0, 
            -1, 
            None
        )
        ngoal = Node(
            self.calc_xyindex(gx, self.minx),
            self.calc_xyindex(gy, self.miny), 
            0.0, 
            -1, 
            None
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart

        while True:
            # Open set is empty
            if len(open_set) == 0:
                break

            # Initialize variables to track the minimum heuristic value and corresponding key
            min_heuristic = float('inf')
            min_key = None

            # Iterate through the items in open_set
            for key, value in open_set.items():
                heuristic_value = self.calc_heuristic(ngoal, value)
                
                # Update minimum heuristic value and corresponding key if a smaller value is found
                if heuristic_value < min_heuristic:
                    min_heuristic = heuristic_value
                    min_key = key

            # Use the key with the minimum heuristic value
            c_id = min_key

            current = open_set[c_id]

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # founded goal
            if current.x == ngoal.x and current.y == ngoal.y:
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2],
                    c_id, 
                    current
                )

                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node) or n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                
                elif open_set[n_id].cost > node.cost:
                    open_set[n_id] = node
                        
        closed_set[ngoal.pind] = current
        rx, ry = self.calc_final_path(ngoal, closed_set)
        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_grid_position(ngoal.x, self.minx)], [self.calc_grid_position(ngoal.y, self.miny)]
        
        n = closedset[ngoal.pind]
        while n is not None:
            rx.append(self.calc_grid_position(n.x, self.minx))
            ry.append(self.calc_grid_position(n.y, self.miny))
            n = n.parent

        return rx[::-1], ry[::-1]

    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, minp):
        """
        calc grid position

        :param index:
        :param minp:
        :return:
        """
        pos = index * self.reso + minp
        return pos

    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.reso)

    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.minx)
        py = self.calc_grid_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        # collision check
        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        self.xwidth = round((self.maxx - self.minx) / self.reso)
        self.ywidth = round((self.maxy - self.miny) / self.reso)
        
        # obstacle map generation
        # Initialize obstacle map with False values
        self.obmap = [[False] * self.ywidth for _ in range(self.xwidth)]

        # Calculate grid positions and mark obstacles
        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.minx)
            
            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.miny)
                
                # Check proximity to obstacles
                if any(math.hypot(iox - x, ioy - y) <= self.rr for iox, ioy in zip(ox, oy)):
                    self.obmap[ix][iy] = True

def main():
    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = .1  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)


    bestfirstsearch = BestFirstSearchPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = bestfirstsearch.planning(sx, sy, gx, gy)
    
    if blender:
        frame = 0 
        robot = bpy.data.objects['M2R2']
        for x,y in zip(rx,ry):
            robot.location.x = x
            robot.location.y = y
            robot.keyframe_insert("location", index=-1, frame=frame)
            frame += 1
    else:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(rx, ry, "-r")
        
        plt.grid(True)
        plt.axis("equal")
        plt.show()

if __name__ == '__main__':
    main()