from vision import Vision
from action import Action
from debug import Debugger
from prm import PRM
from planner import RRT
from pid import PID, check
import time
from zss_debug_pb2 import Debug_Msgs


if __name__ == '__main__':
    vision = Vision()
    action = Action()
    debugger = Debugger()
    # planner = PRM()
    planner = RRT(step_size = 200)
    ctrl = PID()
    
    # 1. path planning & velocity planning
    time.sleep(0.5)
    start_x, start_y = vision.my_robot.x, vision.my_robot.y
    goal_x, goal_y = -2400, -1500
    
    path_x, path_y = planner.plan(vision=vision, start_x=start_x, start_y=start_y, goal_x=goal_x, goal_y=goal_y)
    # 错位绘制点到点连线, 例如1-9=>2-10
    package=Debug_Msgs()
    debugger.draw_lines(
        package, path_x[:-1], path_y[:-1], path_x[1:], path_y[1:])
    debugger.send(package)

    # 总规划点数
    step_index_total = len(path_x) - 1
    step_index = step_index_total
    
    
    while not step_index == 0:
        
        path_x, path_y = planner.plan(vision=vision, start_x=start_x, start_y=start_y, goal_x=goal_x, goal_y=goal_y)
        # 错位绘制点到点连线, 例如1-9=>2-10
        package=Debug_Msgs()
        debugger.draw_lines(
            package, path_x[:-1], path_y[:-1], path_x[1:], path_y[1:])
        debugger.send(package)

        # 总规划点数
        step_index_total = len(path_x) - 1
        step_index = step_index_total
        
        # 信息传入PRM实例planner的规划器plan()
        # 返回值：
        # print("#1 进入控制")
        vx, vw = ctrl.control(vision=vision, path_x=path_x, path_y=path_y, step_index=step_index)

        # 2. send command
        action.sendCommand(vx=vx, vy=0, vw=vw)
        print("vx: ", vx, "vw: ", vw)
        # action.sendCommand(vx=100, vy=0, vw=0)
        # print("now_w #2: ", vision.my_robot.orientation)
        # print("#2 发送控制", "vx: ", vx, "vw:", vw)
        if check(vision.my_robot.x, vision.my_robot.y, path_x[step_index-1], path_y[step_index-1], 30):
            step_index = step_index - 1
        
        time.sleep(0.01)
        
    print("Arrived!")
    action.sendCommand(vx=0, vy=0, vw=0)
    