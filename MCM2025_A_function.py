from tqdm import tqdm
import time

G = 9.8
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from sko.PSO import PSO
import matplotlib.pyplot as plt
import inspect

FY1 = np.array([17800,0,1800])
FY2 = np.array([12000,1400,1400])
FY3 = np.array([6000,-3000,700])
FY4 = np.array([11000,2000,1800])
FY5 = np.array([13000,-2000,1300])
M1 = np.array([20000,0,2000])
M2 = np.array([19000,600,2100])
M3 = np.array([18000,-600,1900])
mod_M1 = np.linalg.norm(M1)
r_cloud = 10
r_aim = 7

def inner_product(x:sp.Matrix):
    return x.dot(x)
# 导数方程定义：
def get_prime():
    x1,y1,z1 = sp.symbols('x1 y1 z1')
    x2,y2,z2 = sp.symbols('x2 y2 z2')
    A = sp.Matrix([x1,y1,z1])
    M = sp.Matrix([x2,y2,z2])
    theta = sp.symbols('theta')
    Real_aim_up = sp.Matrix([r_aim * sp.sin(theta) , r_aim* sp.cos(theta) + 200, 10])
    Real_aim_down = sp.Matrix([r_aim * sp.sin(theta) , r_aim* sp.cos(theta) + 200, 0])

    #最容易得到最大距离的点
    f1_up,f1_down = sp.Matrix.dot(A-M,M-Real_aim_up),inner_product(M-Real_aim_up) # C-V**2/G的形式
    f2_up,f2_down = sp.Matrix.dot(A-M,M-Real_aim_down),inner_product(M-Real_aim_down)
    f1_prime = -(2*f1_up* sp.diff(f1_up,theta) * f1_down - f1_up**2* sp.diff(f1_down,theta))
    f2_prime = -(2*f2_up* sp.diff(f2_up,theta) * f2_down - f2_up**2* sp.diff(f2_down,theta))
    # print(f1_prime)
    # print(f2_prime)
    # critical_points_1 = sp.solve(f1_prime,theta)
    # critical_points_2 = sp.solve(f2_prime,theta)     #得到所有导数为0的theta (算的很慢，需要换函数优化)

    f1_prime = sp.lambdify((theta,x1,y1,z1,x2,y2,z2), f1_prime,'numpy')
    f2_prime = sp.lambdify((theta,x1,y1,z1,x2,y2,z2), f2_prime,'numpy')
    return f1_prime,f2_prime

#遮蔽判断函数
def cover(A:np.array,M:np.array,f1,f2) -> bool:
    def points_vector(A:np.array,M:np.array,N:np.array)->int:  #点到直线的距离（不是向量）
        # A是球心，M是导弹，N是真目标
        return inner_product(A-M) - ((A-M).dot(M-N))**2 / inner_product(M-N)
    def solve_func(func,critical_points):
        initial_guesses = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        sol, info, ier, mesg = fsolve(func, initial_guesses, full_output=True)  #绘制函数图像，观察零点区间（图像类似正弦函数）
        if ier == 1:  # ier == 1 表示求解成功收敛
            for val in sol:
            # 把解限制在 [0, 2*pi] 范围内
                val_normalized = val % (2 * np.pi)
            # 去重存储
                if not any(np.isclose(val_normalized, cp, atol=1e-4) for cp in critical_points):
                    critical_points.append(val_normalized)

    x1,y1,z1 = A[0],A[1],A[2]
    x2,y2,z2 = M[0],M[1],M[2]
    f1_prime = lambda theta: f1(theta,x1,y1,z1,x2,y2,z2)
    f2_prime = lambda theta: f2(theta,x1,y1,z1,x2,y2,z2)
    critical_points_1 = []
    critical_points_2 = []
    solve_func(f1_prime,critical_points_1)
    solve_func(f2_prime,critical_points_2)

    #判断这些点到直线的距离是否小于半径
    for point in critical_points_1:
        N = np.array([r_aim * np.sin(point) , r_aim* np.cos(point) + 200, 10])
        if points_vector(A,M,N) > r_cloud**2: return False
    for point in critical_points_2:
        N = np.array([r_aim * np.sin(point) , r_aim* np.cos(point) + 200, 0])
        if points_vector(A,M,N) > r_cloud**2: return False
    return True


#判断导弹是否通过烟雾只需判断导弹到假目标连线和 烟雾中心、导弹夹角
def out_or_not(A,M)->bool:
    if np.dot(A - M, -M) < 0 and np.dot(A - M,A-M) > r_cloud**2:
    # if np.dot(A - M,A-M) > r_cloud**2:
        return False
    else:
        return True

# 判断这个时间，导弹、烟雾弹相对位置
def check(t0:int,M:np.array,A:np.array,f1_prime,f2_prime) -> int: #A :球心 M：导弹
    missle = M - M/np.linalg.norm(M) * 300 * t0
    smoke = A - np.array([0,0,1]) * 3 * t0
    # 一共有四种相对位置情况
    # 目标：严格保证1234的单调性，其中2为遮蔽，3表示导弹穿出（距离<球半径），1表示烟雾在导弹轨迹上方，4表示烟雾在导弹轨迹下方
    if cover(smoke,missle,f1_prime,f2_prime) == False:
        if smoke[2] / np.sqrt(smoke[0]**2 + smoke[1]**2) > missle[2] / np.sqrt(missle[0]**2 + missle[1]**2): return 1
        else: return 4
    elif cover(smoke,missle,f1_prime,f2_prime) == True and out_or_not(smoke,missle) == True:
        return 2
    elif cover(smoke,missle,f1_prime,f2_prime) == True and out_or_not(smoke,missle) == False:
        return 3

def bisect_find(x,M,A,f1_prime,f2_prime,epsilon_t = 1e-3): #x 表示check函数中的返回值
    t_left,t_right = 0,20
    # for t in np.arange(0,20,0.1):   #验证单调性
    #     print(t, check(t,M,A,f1_prime,f2_prime))

    while t_right - t_left > epsilon_t:
        t_mid = (t_right + t_left) / 2
        if check(t_mid,M,A,f1_prime,f2_prime) >= x:
            t_right = t_mid
        elif check(t_mid,M,A,f1_prime,f2_prime) <= x-1:
            t_left = t_mid
    return t_right

