import numpy as np
from planner import RRT
from vision import Vision
from action import Action
from debug import Debugger
from prm import PRM
import time
import math
from zss_debug_pb2 import Debug_Msgs, Debug_Msg, Debug_Arc

class PID():
    '''
    PID controller for w and v
    
    :param v_kp: v proportion
    :param v_ki: v integration
    :param v_kd: v differentiation
    :param w_kp, w_ki, w_kd: the same for w
    :调参v1: v_kp=800, v_ki=30, v_kd=10, w_kp=30, w_ki=12, w_kd=0.8, 检验距离10/25, 底速280/360
    '''
    def __init__(self, v_kp=800, v_ki=30, v_kd=10, w_kp=30, w_ki=12, w_kd=0.8):
        self.v_kp = v_kp
        self.v_ki = v_ki
        self.v_kd = v_kd
        self.w_kp = w_kp
        self.w_ki = w_ki
        self.w_kd = w_kd
        
        # 增量式PID 当前误差 上一次误差 上上次误差
        self.v_err = [0, 0, 0]
        self.w_err = [0, 0, 0]
        
        # 增量式PID无需考虑积分饱和
        
        # 控制限幅
        # v: 3500, va: 4000, w: 15, wa: 20
        self.w_max = 15
        self.wa_max = 20
        self.v_max = 3500
        self.va_max = 4000
    
    # flage = 'v' => v control
    def pid(self, now, target, flag='v'):
        '''
        返回控制增量
        
        :param now: 当前状态
        :param target: 目标状态
        :param: v or w
        '''
        if flag == 'v':
            self.v_err[0] = target - now
            delta = self.v_kp*(self.v_err[0] - self.v_err[1]) + self.v_ki*self.v_err[0] \
                + self.v_kd*(self.v_err[0] - 2*self.v_err[1] + self.v_err[2])
            
            # 加速度限幅
            if(abs(delta)>=self.va_max):
                if delta > 0:
                    delta = self.va_max
                else:
                    delta = -self.va_max
            
            # 更新误差
            self.v_err[2] = self.v_err[1]
            self.v_err[1] = self.v_err[0]
        elif flag == 'w':
            self.w_err[0] = target - now
            
            # 就近角度调整
            if(abs(self.w_err[0])>np.pi):
                if self.w_err[0]>0:
                    self.w_err[0] -= 2*np.pi
                else:
                    self.w_err[0] += 2*np.pi
            
            delta = self.w_kp*(self.w_err[0] - self.w_err[1]) + self.w_ki*self.w_err[0] \
                + self.w_kd*(self.w_err[0] - 2*self.w_err[1] + self.w_err[2])
            
            # 角加速度限幅
            if(abs(delta)>=self.wa_max):
                if delta > 0:
                    delta = self.wa_max
                else:
                    delta = -self.wa_max
            
            # 更新误差
            self.w_err[2] = self.w_err[1]
            self.w_err[1] = self.w_err[0]
        
        return delta
    
    # 控制完成需检验当前位置 确定step_index位置
    def control(self,  vision, path_x, path_y, step_index, num=0, forward=True) -> tuple[int, int] :
        '''
        master controller
        
        :param: vision: 全局视觉
        :param: path_x, path_y: 通往目标路径
        :param: step_index: 当前执行路径序号
        '''
        # 序号前进方向
        if forward:
            index_dir = -1
        else:
            index_dir = 1
        
        my_robot = vision.blue_robot[num]
        
        now_w = my_robot.orientation
        now_v = my_robot.vel_x
        now_x = my_robot.x
        now_y = my_robot.y
        
        target_x = path_x[step_index+index_dir]
        target_y = path_y[step_index+index_dir]
        # 为计算简便，以到达目标点的距离为期望速度，角度差为期望角速度
        target_w = angle(now_x, now_y, target_x, target_y)
        target_v = dist(now_x, now_y, target_x, target_y)
        
        # print('target_angle: ',target_w)
        # print('now_w', now_w)
        # print('dist: ', target_v)
        # print('now_v: ', now_v)
        
        delta_w = self.pid(now_w, target_w, 'w')
        delta_v = self.pid(now_v, target_v, 'v')
        
        vw = delta_w
        # print("delta_w: ", delta_w)
        vx = now_v + delta_v
        
        # 角速度和速度限幅
        if(abs(vw)>self.w_max):
            if vw>0:
                vw = self.w_max
            else:
                vw = -self.w_max
            
        if abs(vx)>self.v_max:
            vx = self.v_max
        elif vx<0:
            vx = 0
        
        # 防止角速度过大就飞出
        if abs(vw) >= 0.1:
            vx = 360

        return vx, vw

        
def angle(x0, y0, x1, y1) -> float:
        '''
        计算角度并转范围到[-pi, pi), 正左方为-pi, 逆时针转动为正
        '''
        x = x1 - x0
        y = y1 - y0
        
        # arctan2范围(-pi, pi], 正上方为0
        angle = np.arctan2(x, y)
        
        if angle < -np.pi/2:
            angle = -angle - np.pi*3/2
        elif angle == np.pi:
            angle = -np.pi/2
        else:
            angle = -angle + np.pi/2

        return angle
    
def dist(x0, y0, x1, y1) -> int:
    '''
    计算距离
    '''
    x = x1 - x0
    y = y1 - y0
    
    dist = np.sqrt(np.square(x) + np.square(y))
    
    return dist
    
def check(x0, y0, x1, y1, min_dist=100) -> bool:
        '''
        检查是否到达下一节点
        '''
        _dist = dist(x0, y0, x1, y1)
        
        if _dist<min_dist:
            return True
        else:
            return False