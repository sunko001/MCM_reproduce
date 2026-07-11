from MCM2025_A_function import *

def func2_function(x, f1_prime, f2_prime):
    v, theta, t_straight, t_drop = x
    M1_ps = M1 - M1 / mod_M1 * 300 * (t_straight + t_drop)
    A_circle = FY1 + np.array([v * np.cos(theta), v * np.sin(theta), 0]) * (t_straight + t_drop) - 0.5 * G * np.array(
        [0, 0, 1]) * t_drop ** 2
    left = bisect_find(2, M1_ps, A_circle, f1_prime, f2_prime)
    right = bisect_find(3, M1_ps, A_circle, f1_prime, f2_prime)
    return -(right - left)


# 第二问
def func2(f1_prime, f2_prime):
    """
    飞行方向，飞行速度，投放时刻，引爆时刻，四个自由度
    我们只需要烟雾弹引爆时刻的坐标，对应此时导弹的坐标（投放、引爆时间）
    实际上，上述自由度导致了引爆点在一个空间中
    :return:
    """

    # 时间包含平抛时间、平移时间
    # 最长整个时间不能超过导弹打到目标
    t_max = mod_M1 / 300
    # 平移时间上限
    t_straight_max = (M1[2] - FY1[2] + r_cloud) / (300 * np.sin(np.arctan(2000 / 20000)))  # 这个是导弹飞行到烟雾弹上方的时间
    # 平抛时间不能太长，使得烟雾弹落体到地
    t_down_max = np.sqrt(2 * FY1[2] / G)
    # 角度约束：不能平行y轴-> x正负半轴附近
    # 讲评中指出了平行y负半轴飞慢点好：这个情况下遮蔽的时候z更小，留给无人机的时间充足；（个人感觉这两个速度约束没有很好的条件、方法实现）因此此处不对速度70-140进行限制
    # theta偏向角用和真目标连线的夹角求得，arctan(200/17800) (可以乘以一个倍数扩大一下)
    # 用智能优化算法实现：以PSO为例
    start_time = time.perf_counter()
    # theta_max_1 = np.arctan(200/17800) * 2 # 这个是不对的(有比它更大的情况)（乘以一个大点的倍率也行，如5倍、10倍）
    theta_max_1 = np.pi  # 两种情况统一
    lb = [70, 0, 0, 0]
    ub = [140, theta_max_1, t_straight_max, t_down_max]
    func2_func = lambda x: func2_function(x, f1_prime, f2_prime)
    pso_advanced = PSO(func2_func, 4, pop=100, max_iter=50, lb=lb, ub=ub, w=0.8)

    # pso_advanced.run()
    # best_x = pso_advanced.gbest_x
    # best_y = pso_advanced.gbest_y
    # print(f"最优参数: {best_x}, 最大遮蔽时间: {-best_y:.6f} 秒, 总耗时: {time.perf_counter() - start_time:.6f} 秒")

    history = []
    for i in range(pso_advanced.max_iter):

        pso_advanced.run(1)
        best_x = pso_advanced.gbest_x
        best_y = -pso_advanced.gbest_y
        history.append(best_y)

        elapsed = time.perf_counter() - start_time
        if i == 0 or (i + 1) % 10 == 0:
            print(
                f"第{i + 1:3d}代 | "
                f"飞行速度: {best_x[0]:.2f} | " f"飞行角度: {best_x[1]:.4f} rad | " f"平移时间: {best_x[2]:.4f} s | " f"平抛时间: {best_x[3]:.4f} s | "
                f"当前最优: {best_y:.6f} s | "
                f"耗时: {elapsed:.2f} s | "
            )
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Max Cover Time (s)")
    plt.title("PSO Optimization Progress")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    f1, f2 = get_prime()

    # 第二问验证一个解的测试代码
    # v,theta,t_straight,t_drop = 136.3663,0.0902,0.8977,0.0497
    # M1_ps = M1 - M1 / mod_M1 * 300 * (t_straight + t_drop)
    # A_circle = FY1 + np.array([v*np.cos(theta),v*np.sin(theta),0]) * (t_straight + t_drop) - 0.5 * G * t_drop**2 * np.array([0,0,1])
    # print(f"导弹位置: {M1_ps}, 烟雾弹位置: {A_circle}")
    # left = bisect_find(2,M1_ps,A_circle,f1_prime,f2_prime)
    # right = bisect_find(3,M1_ps,A_circle,f1_prime,f2_prime)
    # print(M1_ps,A_circle)
    # print(left,right)
    # print(f"最大遮蔽时间: {right - left:.6f} 秒")

    func2(f1,f2)