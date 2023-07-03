import random
import math
import time
from abc import ABC, abstractmethod
import numpy as np
from action import Action
from debug import Debugger
from vision import Vision
from zss_debug_pb2 import Debug_Msgs


class Planner(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def plan(self, vision: Vision, start: int, start_y: int, goal_x: int, goal_y: int) -> tuple[list[int], list[int]]:
        '''
        generate a path from the starting point to the goal point based on the situation of the game

        :param vision: description of venue conditions
        :param start_x: the x-coordinate of the starting point
        :param start_y: the x-coordinate of the starting point
        :param goal_x: the x-coordinate of the goal point
        :param goal_y: the y-coordinate of the goal point
        :returns    path_x: list of x-coordinate of the path planning points, 
                    path_y: list of y-coordinate of the path planning points
        '''

        return [], []


class RRT(Planner):
    def __init__(self, max_sample: int = 1000, step_size: int = 200, max_steer_ang: float = math.pi / 4, goal_prob: float = 0.1) -> None:
        '''
        init RRT planner

        :param max_sample: max number of sample points
        :param step_size: forward distance to move at one time
        :param max_steer_ang: max steer angle to move at one time
        :param goal_prob: probability of sampling to the target point, [0, 1)
        '''

        super().__init__()
        self.max_sample = max_sample
        self.step_size = step_size
        self.max_steer_ang = max_steer_ang
        self.goal_prob = goal_prob
        self.minx = -4500
        self.maxx = 4500
        self.miny = -3000
        self.maxy = 3000
        self.robot_size = 200
        self.avoid_dist = 200
        self.goal_dist_thres = 200

    def plan(self, vision: Vision, start_x: int, start_y: int, goal_x: int, goal_y: int) -> tuple[list[int], list[int]]:
        '''
        use RRT algorithm to generate a path from the starting point to the goal point based on the situation of the game

        :param vision: description of venue conditions
        :param start_x: the x-coordinate of the starting point
        :param start_y: the x-coordinate of the starting point
        :param goal_x: the x-coordinate of the goal point
        :param goal_y: the y-coordinate of the goal point
        :returns    path_x: list of x-coordinate of the path planning points, 
                    path_y: list of y-coordinate of the path planning points
        '''

        # RRT tree
        rrt_node_x = [start_x]
        rrt_node_y = [start_y]
        rrt_parent_index = [-1]

        # get obstacles' position
        obs_x = []
        obs_y = []
        for robot in vision.blue_robot[1:]:
            if robot.visible:
                obs_x.append(robot.x)
                obs_y.append(robot.y)
        for robot in vision.yellow_robot:
            if robot.visible:
                obs_x.append(robot.x)
                obs_y.append(robot.y)

        for _ in range(self.max_sample):
            # sample
            if random.random() < self.goal_prob:
                # with a certain probability of directly sampling to the goal point
                pnt_rand_x, pnt_rand_y = goal_x, goal_y
            else:
                pnt_rand_x, pnt_rand_y = self._sample()

            # near
            pnt_index = self._near(
                pnt_rand_x, pnt_rand_y, rrt_node_x, rrt_node_y)

            # steer
            parent_index = rrt_parent_index[pnt_index]
            if parent_index == -1:
                pnt_near_ang = vision.my_robot.orientation
            else:
                pnt_near_ang = math.atan2(rrt_node_y[pnt_index] - rrt_node_y[parent_index],
                                          rrt_node_x[pnt_index] - rrt_node_x[parent_index])
            pnt_new_x, pnt_new_y = self._steer(pnt_rand_x, pnt_rand_y,
                                               rrt_node_x[pnt_index], rrt_node_y[pnt_index], pnt_near_ang)

            # collision check
            if not self._check_collision(
                    obs_x, obs_y, rrt_node_x[pnt_index], rrt_node_y[pnt_index], pnt_new_x, pnt_new_y):
                rrt_node_x.append(pnt_new_x)
                rrt_node_y.append(pnt_new_y)
                rrt_parent_index.append(pnt_index)

            # arrival check
            if math.sqrt((goal_x - pnt_new_x)**2 + (goal_y - pnt_new_y)**2) < self.goal_dist_thres:
                break
        else:
            print("Not find path to goal!")
            return [], []

        # generate path
        path_x = []
        path_y = []
        tmp_i = len(rrt_parent_index) - 1
        while tmp_i != -1:
            path_x.append(rrt_node_x[tmp_i])
            path_y.append(rrt_node_y[tmp_i])
            tmp_i = rrt_parent_index[tmp_i]
        print("Find path to goal!")

        return path_x, path_y

    def _sample(self) -> tuple[float, float]:
        '''
        sample in a map range to obtain a random point

        :returns    pnt_rand_x: the x-coordinate of the sampling point
                    pnt_rand_y: the y-coordinate of the sampling point
        '''

        pnt_rand_x = (random.random() * (self.maxx - self.minx)) + self.minx
        pnt_rand_y = (random.random() * (self.maxy - self.miny)) + self.miny

        return pnt_rand_x, pnt_rand_y

    def _near(self, pnt_rand_x: float, pnt_rand_y: float, rrt_node_x: list[int], rrt_node_y: list[int]) -> int:
        '''
        find index of the nearest node to the sampling point in RRT tree

        :param pnt_rand_x: the x-coordinate of the sampling point
        :param pnt_rand_y: the y-coordinate of the sampling point
        :param rrt_node_x: list of the x-coordinate of the RRT tree points
        :param rrt_node_y: list of the y-coordinate of the RRT tree points
        :returns    pnt_index: the index of the nearest point to the sampling point in RRT tree
        '''

        nearest_dist = np.inf
        pnt_index = -1
        for i, (x, y) in enumerate(zip(rrt_node_x, rrt_node_y)):
            dist = math.sqrt((pnt_rand_x - x)**2 + (pnt_rand_y - y)**2)
            if dist < nearest_dist:
                nearest_dist = dist
                pnt_index = i

        return pnt_index

    def _steer(self, pnt_rand_x: int, pnt_rand_y: int, pnt_near_x: int, pnt_near_y: int, pnt_near_ang: float) -> tuple[int, int]:
        '''
        generate new node

        :param pnt_rand_x: the x-coordinate of the sampling point
        :param pnt_rand_y: the y-coordinate of the sampling point
        :param pnt_near_x: the x-coordinate of the nearest point to the sampling point
        :param pnt_near_y: the y-coordinate of the nearest point to the sampling point
        :param pnt_near_ang: the orientation of the nearest point to the sampling point
        :returns    pnt_new_x: the x-coordinate of the new point
                    pnt_new_y: the y-coordinate of the new point
        '''

        move_ang = math.atan2(pnt_rand_y - pnt_near_y, pnt_rand_x - pnt_near_x)
        max_ang = pnt_near_ang + self.max_steer_ang
        min_ang = pnt_near_ang - self.max_steer_ang
        if abs(max_ang - move_ang) > math.pi:
            if move_ang > 0 and max_ang < 0:
                max_ang += 2 * math.pi
            elif move_ang < 0 and max_ang > 0:
                max_ang -= 2 * math.pi
        if abs(min_ang - move_ang) > math.pi:
            if move_ang > 0 and min_ang < 0:
                min_ang += 2 * math.pi
            elif move_ang < 0 and min_ang > 0:
                min_ang -= 2 * math.pi

        # limit steer angle
        if move_ang > max_ang:
            move_ang = max_ang
        elif move_ang < min_ang:
            move_ang = min_ang

        # generate new point
        pnt_new_x = math.cos(move_ang) * self.step_size + pnt_near_x
        pnt_new_y = math.sin(move_ang) * self.step_size + pnt_near_y

        return pnt_new_x, pnt_new_y

    def _check_collision(
            self, obs_x: list[int], obs_y: list[int], pnt_near_x: int, pnt_near_y: int, pnt_new_x: int, pnt_new_y: int) -> bool:
        '''
        check if the movement to the next point collide with an obstacle

        :param obs_x: list of the x-coordinate of the obstacle
        :param obs_y: list of the y-coordinate of the obstacle
        :param pnt_near_x: the x-coordinate of the nearest point to the sampling point
        :param pnt_near_y: the y-coordinate of the nearest point to the sampling point
        :param pnt_new_x: the x-coordinate of the new point
        :param pnt_new_y: the y-coordinate of the new point
        :returns is_collided: return true when the movement will cause collision, otherwise return false
        '''

        move_vec_x = pnt_new_x - pnt_near_x
        move_vec_y = pnt_new_y - pnt_near_y
        move_vec_len = math.sqrt(move_vec_x**2 + move_vec_y**2)

        for x, y in zip(obs_x, obs_y):

            # calculate the vertical distance from the obstacle to the moving line
            obs_vec_x = x - pnt_near_x
            obs_vec_y = y - pnt_near_y
            dist_v = abs(obs_vec_x * move_vec_y -
                         obs_vec_y * move_vec_x) / move_vec_len

            if dist_v < self.robot_size / 2 + self.avoid_dist:

                # calculate the projected distance from the obstacle to the moving line
                dist_p = (obs_vec_x * move_vec_x +
                          obs_vec_y * move_vec_y) / move_vec_len

                if dist_p > 0:

                    # calculate the movement distance required for a collision to occur in a predetermined direction of movement
                    ddist_p = math.sqrt(
                        (self.robot_size / 2 + self.avoid_dist)**2 - dist_v**2)
                    if dist_p - ddist_p < move_vec_len:
                        return True
        else:
            return False


if __name__ == "__main__":
    planner = RRT()
    vision = Vision()
    action = Action()
    debugger = Debugger()
    time.sleep(0.1)

    while True:
        # 1. path planning & velocity planning
        start_x, start_y = vision.my_robot.x, vision.my_robot.y
        goal_x, goal_y = -2400, -1500
        path_x, path_y = planner.plan(vision=vision,
                                      start_x=start_x, start_y=start_y, goal_x=goal_x, goal_y=goal_y)

        # 2. send command
        action.sendCommand(vx=0, vy=0, vw=0)

        # 3. draw debug msg
        package = Debug_Msgs()
        # 错位绘制点到点连线, 例如1-9=>2-10
        debugger.draw_lines(
            package, path_x[:-1], path_y[:-1], path_x[1:], path_y[1:])
        debugger.send(package)

        time.sleep(0.01)
