from MCM2025_A_function import *

#第一问
def func1():
    f1_prime,f2_prime = get_prime()
    # print(inspect.getsource(f1_prime))
    # print(inspect.getsource(f2_prime))

    start = time.time()
    ans = 0
    #无人机投放点位置
    FY1_drop = FY1 - np.array([1,0,0]) * 120 * 1.5
    FY1_boom = FY1_drop - np.array([1,0,0]) * 120 * 3.6 - np.array([0,0,1]) * 0.5 * G * 3.6**2
    Missile_position = M1 - M1/np.linalg.norm(M1) *300 *(1.5 + 3.6)
    # print(Missile_position,FY1_boom)
    # for delt_t in np.arange(0,20,0.01):
    #     y_cloud = FY1_boom - np.array([0,0,3]) * delt_t
    #     Missile_point = Missile_position - M1/np.linalg.norm(M1) * delt_t * 300
    #     if cover(y_cloud,Missile_point):
    #         # 判断导弹是否穿过云团
    #         if out_or_not(y_cloud,Missile_point):
    #             ans += 0.01

    # 可利用二分法大大降低算法复杂度
    # 二分查找开始遮蔽的时间（bisect_left） + 二分查找遮蔽结束的时间(需改变单调性，再bisect_left)
    # 讲评中改进二分法思路：
    # 先查找开始遮蔽的时刻（此处以烟雾弹爆炸时刻为0时刻）
    # 找到第一个return 为2的
    # deleta_t = np.arange(0,20,1e-4)
    # left = bisect.bisect_left(deleta_t,2,key= lambda t:check(t,Missile_position,FY1_boom,f1_prime,f2_prime))
    # right = bisect.bisect_left(deleta_t,3,key= lambda t:check(t,Missile_position,FY1_boom,f1_prime,f2_prime))
    # ans = deleta_t[right] - deleta_t[left]
    # print(ans)

    ## 如何获取更高的精度：把方程零点放到哈希表，每个A，M都对应一个极值点theta（防止多次调用fslove）
    # 手写二分搜索
    left = bisect_find(2,Missile_position,FY1_boom,f1_prime,f2_prime)
    right = bisect_find(3,Missile_position,FY1_boom,f1_prime,f2_prime)
    print(f"开始遮蔽时间 {left}")
    print(f"结束遮蔽时间 {right}")
    print(f"遮蔽时长 {right-left}")
    print(f"单次目标函数计算耗时: {time.time() - start:.4f} 秒")

if __name__ == "__main__":
    func1()