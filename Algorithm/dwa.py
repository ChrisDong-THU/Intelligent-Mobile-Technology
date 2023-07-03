from vision import Vision
from action import Action
from debug import Debugger
from zss_debug_pb2 import Debug_Msgs

from pid import PID

import numpy as np
import math
import random
import time

class DWA():
    '''DWA
    
    DWA implimentation
    
    '''
    def __init__(self, vision, action, debugger, goal, is_debug=False) -> None:
        self._vision = vision
        self._action = action
        self._debugger = debugger
        self._is_debug = is_debug
        
        self._vel = 0 # 不是实时速度, 用于设定
        self._w = 0 # 无法获取机器人的实时角速度，也只用于设定
        self._traj = np.array([])
        
        self._vel_max = 2000
        self._w_max = 20
        self._dv = 600 # 速度采样分度
        self._dw = 15
        self._v_acc_max = 4000
        self._w_acc_max = 20
        self._v_samples = 8
        self._w_samples = 20
        
        self._kvw = [12, 0.01] # 速度与角速度变化代价折算系数
        self._predict_time = 1.6
        self._predict_dt = 0.2 #采样时间分度
        
        self._radius = 120 # 机器人半径
        self._checkdist = 100 # 检查到终点距离
        self._checkdist_pid = 1000
        self._checkdist_toward = 2000 #进入循向范围
        self._goal = goal
        self._obs = [] # 障碍物坐标二维数组
        
        # cost计算系数 goal, vel, obs
        self._alpha = 0.1
        self._beta = 1
        self._gamma = 10
 
    @property
    def vel(self) -> int:
        # 坐标获取通信延迟问题
        return abs(self._vision.my_robot.vel_x)
    @property
    def x(self) -> int:
        return self._vision.my_robot.x
    @property
    def y(self) -> int:
        return self._vision.my_robot.y
    @property
    def ori(self) -> float:
        
        return self._vision.my_robot.orientation
    # 设定值暂存_v和_w中
    def vwset(self, vel, w) -> None:
        delta_v = self.vel-vel
        if abs(delta_v) > self._v_acc_max:
            # print("Chang vel too fast!")
            self._vel += self._v_acc_max*self.sign(delta_v)
        else:
            self._vel = vel
        self._vel = min(self._vel_max, max(self._vel, -200))
        delta_w = self._w-w
        if abs(delta_w) > self._w_acc_max:
            # print("Change w too fast!")
            self._w += self._w_acc_max*self.sign(delta_w)
        else:
            self._w = w      
        self._w = max(-self._w_max, min(self._w_max, self._w))
        
    @staticmethod
    def sign(num) -> int:
        '''
        正数返回1
        '''
        if num>0: return -1
        elif num<0: return 1
        
    # 距离目标点评价函数
    @staticmethod
    def goal_cost(goal, traj, isTorward=False):
        '''目标点距离评价
        Args:
            goal(ndarray): (2,)
            traj(ndarray): (n, 5)
        Returns:
            cost
        '''
        # print("goal: ", goal[0], goal[1])
        # print("traj: ", traj[-1,0], traj[-1,1])
        if isTorward==False:
            k=1
        else:
            k=10

        return math.sqrt((goal[0]-traj[-1,0])**2+(goal[1]-traj[-1,1])**2)*k
    
    # 速度评价函数1
    @staticmethod
    def vel_cost(traj, last_v, last_w, kvw):
        '''速度评价函数1

        Args:
            traj (ndarray): n点轨迹
            last_v (int): 上一轨迹速度
            last_w (float): 上一轨迹角速度
            kvw (float): 速度角速度代价换算系数

        Returns:
            float: 总的速度评价
        '''
        # 取最后一点速度作为速度代价计算依据（实际各点速度相同）
        vel = traj[-1, 3]
        w = traj[-1, 4]
        
        cost_v = abs(vel - last_v)
        cost_w = abs(w - last_w)

        return cost_w*kvw[0] + cost_v*kvw[1]
    
    # 速度评价函数2, 作备用, 整个轨迹的加速度变化
    def vel_cost_backup(self, traj):
        '''速度评价2: 轨迹内各点速度仍在改变
        
        评价轨迹中速度采样的易操作性
        
        Args:
            traj(ndarray): (n, 5)
        Returns:
            cost
        '''
        steps = traj.shape[0]
        # 速度集和角速度集
        vels = traj[:, 3]
        ws = traj[:, 4]
        
        delta_vels = vels[1:] - vels[:-1]
        delta_ws = ws[1:] - ws[:-1]
        
        cost_v = sum(abs(delta_vels))
        cost_w = sum(abs(delta_ws))
        
        return cost_w + self._kvw*cost_v
    
    # 距离障碍物评价函数
    @staticmethod
    def obs_cost(traj, obs, r):
        '''障碍物距离
        
        找离障碍物的最小距离
        
        Args:
            traj(ndarray): (n, 4)轨迹
            obs(ndarray): (m, 2)障碍物坐标
            r: 安全距离
        '''
        min_dist = float('Inf') # 初始化无穷大距离
        
        for i in range(traj.shape[0]):
            for j in range(len(obs)):
                current_dist = math.sqrt((traj[i, 0]-obs[j][0])**2+(traj[i, 1]-obs[j][1])**2)
                if current_dist < r:
                    # print("触发障碍物检测！")
                    return float('Inf')
                if current_dist < min_dist:
                    min_dist = current_dist
                    
        return 1/min_dist
        
    # 速度空间内采样
    @staticmethod
    def v_sample(current_v, dv) -> int:
        current_v += (2*random.random()-1)*dv
             
        return current_v
    @staticmethod
    def w_sample(current_w, dw) -> float:
        current_w += (2*random.random()-1)*dw
        
        return current_w
    
    # 一个dt内模拟一个速度的轨迹
    @staticmethod
    def onestep_calc(state, v, w, dt):
        '''
        Args:
            ori: 当前车朝向
            state: 一个点的信息(5,), 改变后作为返回的下一个点的信息
            dt: 预测时间分度
        Returns:
            state: 下一状态, x, y, ori, v, w
        '''
        state[0] += state[3]*dt*math.cos(state[2])
        state[1] += state[3]*dt*math.sin(state[2])
        state[2] += state[4]*dt
        
        state[3] = v
        state[4] = w
        
        return state

    # 一个predict_time内模拟速度的轨迹
    def onetraj_calc(self, state, v, w):
        '''
        Args:
            state: 初始状态(5,)
            v: 轨迹基于的采样速度
            w: 轨迹基于的采样角速度
        Returns:
            traj: 单条轨迹
        '''
        traj = np.array(state)
        newstate = np.array(state)
        t = 0
        
        # 轨迹长度由predict_time决定
        while t <= self._predict_time:
            newstate = self.onestep_calc(newstate, v, w, self._predict_dt)
            traj = np.vstack((traj, newstate))
            
            t += self._predict_dt
            
        return traj
    
    # obstacles建图
    def scan_obs(self):
        self._obs.clear()
        
        for robot_blue in self._vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                # 非本机且可视加入障碍物机器人列表
                # 分别保存障碍物机器人的坐标x和y
                self._obs.append((robot_blue.x, robot_blue.y))
        for robot_yellow in self._vision.yellow_robot:
            if robot_yellow.visible:
                self._obs.append((robot_yellow.x, robot_yellow.y))
    
    # DWA核心
    def dwa_core(self, state, isToward=False):
        self.scan_obs() # 扫描一次障碍物
        
        min_cost = 10000.0
        flag = 0
        vsample = state[3]
        dv = self._dv
        # 采样区间上抬和速度平滑缓冲
        if vsample<1500 and not isToward:
            vsample=vsample*0.4+900
        if vsample<1200 and not isToward:
            vsample=vsample*0.4+720
        if isToward:
            vsample = max(300+vsample*0.2, vsample)
            if vsample<200:
                vsample=vsample*0.5+100
            dv = dv*0.8
        
        for i in range(self._v_samples):
            v = self.v_sample(vsample, dv)
            v = min(self._vel_max, max(80, v))

            for j in range(self._w_samples):
                dw = self._dw
                if v>800:
                    dw = dw*0.4
                # if isToward:
                #     dw = dw*1.4
                w = self.w_sample(0, dw)
                
                traj = self.onetraj_calc(state, v, w)
                
                goal_cost = self.goal_cost(self._goal, traj, isToward)*self._alpha
                vel_cost = self.vel_cost(traj, state[3], state[4], self._kvw)*self._beta
                obs_cost = self.obs_cost(traj, self._obs, self._radius*2)*self._gamma
                
                total_cost = goal_cost + vel_cost + obs_cost
                
                if min_cost >= total_cost:
                    min_cost = total_cost
                    best_traj = traj
                    flag = 1
        
        if flag == 0:
            best_traj = np.zeros((2, 7))
            best_traj[1, 3] = 240
            best_traj[1, 4] = self.w_sample(0, self._dw*0.1)
        
        print("最佳速度、角速度：", best_traj[1, 3], best_traj[1,4])
        self.drawtraj(self._debugger, best_traj)
                    
        return best_traj
    
    # traj: 长度为 pred_time/pred_dt+2
    @staticmethod
    def drawtraj(debugger, traj):
        traj_x = [traj[i, 0] for i in range(traj.shape[0])]
        traj_y = [traj[i, 1] for i in range(traj.shape[0])]
        
        package=Debug_Msgs()
        debugger.draw_lines(
            package, traj_x[:-1], traj_y[:-1], traj_x[1:], traj_y[1:]
        )
        debugger.send(package)
    
    # 单步执行
    def runstep(self, v, w, isToward=False):
        # 执行动作
        self.vwset(v, w)
        self._action.sendCommand(vx=self._vel, vy=0, vw=self._w)
        # 考虑程序运行本身延时
        time.sleep(self._predict_dt*0.1)
        # time.sleep(0.001)
        
        # 读取状态
        state = [self.x, self.y, self.ori, self.vel, 0]
        traj = self.dwa_core(state, isToward)
        
        # 取新轨迹的第1步为下一次执行
        new_v = traj[1, 3]
        new_w = traj[1, 4]
        
        return new_v, new_w
        
    def checkgoal(self, checkdist):
        if math.sqrt((self.x-self._goal[0])**2+\
            (self.y-self._goal[1])**2) <= checkdist:
            return True
        else:
            return False
    
    # 来回时变换目的地
    def turnaround(self):
        self._goal[0] = -self._goal[0]
        self._goal[1] = -self._goal[1]

    
if __name__ == '__main__':
    vision = Vision()
    action = Action()
    debugger = Debugger()
    ctrl = PID()
    ctrl.v_max = 1000
    goal = [-2400, -1500]
    dwa = DWA(vision, action, debugger, goal)
    
    v, w=(0, 0)
    
    for i in range(10):
        while not dwa.checkgoal(dwa._checkdist_toward):
            v, w = dwa.runstep(v, w)
        while not dwa.checkgoal(dwa._checkdist_pid):
            v, w = dwa.runstep(v, w, isToward=True)
        while not dwa.checkgoal(dwa._checkdist):
            v, w = ctrl.control(vision=dwa._vision, path_x=[dwa._goal[0]], \
                path_y=[dwa._goal[1]], step_index=1)
            package=Debug_Msgs()
            dwa._debugger.draw_lines(
                package, [dwa.x], [dwa.y], [dwa._goal[0]], [dwa._goal[1]]
            )
            dwa._debugger.send(package)
            dwa._action.sendCommand(v, 0, w)
            time.sleep(0.01)
            
        dwa._action.sendCommand(0, 0, 0)
        dwa.turnaround()
    
    dwa._action.sendCommand(0, 0, 0)