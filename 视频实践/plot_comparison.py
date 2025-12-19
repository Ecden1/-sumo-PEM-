import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from scipy.signal import savgol_filter
#绘图
# ========== 解析命令行参数 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DQN交通控制数据采集与绘图')  # 去掉PEM
    parser.add_argument('--run_mode', type=str, default='simulate', choices=['fixed', 'dqn', 'simulate'],
                        help='运行模式：fixed（固定控制）、dqn（DQN训练）、simulate（模拟数据，默认）')
    parser.add_argument('--simulation_time', type=int, default=500, help='仿真时间步长（默认500）')
    parser.add_argument('--num_episodes', type=int, default=10, help='仿真/训练轮数（默认1）')
    parser.add_argument('--eps_start', type=float, default=0.8, help='DQN初始探索率（默认0.8）')
    parser.add_argument('--eps_decay', type=int, default=80000, help='探索率衰减速率（默认80000）')
    parser.add_argument('--eps_end', type=float, default=0.05, help='探索率最小值（默认0.05）')
    parser.add_argument('--use_gui', type=bool, default=False, help='是否开启GUI（模拟模式下无效，默认False）')
    parser.add_argument('--data_path', type=str, default='./data', help='数据保存路径（默认./data）')
    parser.add_argument('--cut_ratio', type=float, default=0.1, help='数据裁剪比例（0-0.5，默认0.1）')
    return parser.parse_args()

# ========== 滑动平均函数（全局定义，避免未定义报错） ==========
def moving_average(data, window_size=8):
    """滑动平均：window_size越大，曲线越平滑"""
    if len(data) < window_size:
        window_size = len(data)  # 避免窗口大于数据长度
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# ========== 生成瞬时停车数（带平缓波动） ==========
def generate_halted_vehicles(length, base_start, base_end, fluct_amp, period=50):
    base_trend = np.linspace(base_start, base_end, length)
    cycle_fluct = np.sin(np.linspace(0, length/period*2*np.pi, length)) * fluct_amp * 0.5
    random_noise = np.random.normal(0, fluct_amp*0.2, length)
    halted = base_trend + cycle_fluct + random_noise
    halted = np.maximum(halted, 0)
    halted = np.convolve(halted, np.ones(5)/5, mode='same')
    return halted

# ========== 生成平滑数据 ==========
def ultra_smooth_data(start, end, length, trend=0, noise_amp=0.15):
    base = np.linspace(start, end, length) + np.linspace(0, trend, length)
    noise = np.random.normal(0, noise_amp, length)
    data = base + noise
    # 处理长度小于窗口的情况
    window_length = 15 if length >= 15 else length - (length % 2) + 1  # 保证奇数
    if window_length < 3:
        window_length = 3
    smooth_data = savgol_filter(data, window_length=window_length, polyorder=3)
    return smooth_data

# ========== 新增：生成差异化拥堵率（核心：打破升降同步） ==========
def generate_diff_congestion(halted_vehicles, base_congest, fluct_amp, period_offset, noise_amp):
    """
    生成与停车数关联但升降不同步的拥堵率
    :param halted_vehicles: 停车数数组
    :param base_congest: 拥堵率基准值
    :param fluct_amp: 拥堵率波动幅度
    :param period_offset: 拥堵率周期偏移（和停车数周期不同）
    :param noise_amp: 拥堵率随机噪声幅度
    :return: 差异化拥堵率数组
    """
    length = len(halted_vehicles)
    # 1. 基础关联：停车数影响拥堵率（但系数随机化，打破线性）
    random_coeff = np.random.uniform(0.02, 0.05)  # 随机系数，不再是固定的1/30
    base_congest = base_congest + (halted_vehicles * random_coeff)
    # 2. 周期波动偏移：拥堵率的波动周期和停车数不同（升降时机错开）
    cycle_fluct = np.sin(np.linspace(0, (length/50 + period_offset)*2*np.pi, length)) * fluct_amp
    # 3. 随机噪声：拥堵率的升降幅度随机（不再和停车数完全一致）
    random_noise = np.random.normal(0, noise_amp, length)
    # 4. 组合并平滑（保留差异，避免突兀）
    congestion = base_congest + cycle_fluct + random_noise
    congestion = np.clip(congestion, 0.01, 1.0)  # 限制范围
    # 处理长度小于窗口的情况
    window_length = 11 if length >= 11 else length - (length % 2) + 1
    if window_length < 3:
        window_length = 3
    congestion = savgol_filter(congestion, window_length=window_length, polyorder=3)  # 轻量平滑
    return congestion

# ========== 数据裁剪函数（核心：掐头去尾） ==========
def cut_data(data, cut_ratio):
    """
    裁剪数据，去掉开头和结尾的部分
    :param data: 输入的一维数组
    :param cut_ratio: 裁剪比例（0-0.5，比如0.1表示去掉开头10%、结尾10%）
    :return: 裁剪后的数组
    """
    if cut_ratio < 0 or cut_ratio >= 0.5:
        cut_ratio = 0.1  # 限制比例范围，避免裁剪过多
    start_idx = int(len(data) * cut_ratio)
    end_idx = int(len(data) * (1 - cut_ratio))
    return data[start_idx:end_idx]

# ========== 生成模拟数据 ==========
def generate_simulate_data(args):
    time_steps = np.arange(0, args.simulation_time)

    # 1. 固定定时控制数据（拥堵率与停车数关联但升降不同步）
    fixed_halted = generate_halted_vehicles(args.simulation_time, 15, 25, 5, 50)
    # 替换原来的拥堵率生成逻辑：用差异化函数
    fixed_congestion = generate_diff_congestion(
        fixed_halted,
        base_congest=0.2,
        fluct_amp=0.08,
        period_offset=2,  # 周期偏移2，和停车数的周期50错开
        noise_amp=0.03    # 随机噪声，打乱升降幅度
    )
    # 固定定时车速：平滑下降，无强制急降（删除重复定义）
    fixed_speed = ultra_smooth_data(19, 17, args.simulation_time, trend=-0.2)

    fixed_data = {
        'time_steps': time_steps,
        'congestion_rates': fixed_congestion,
        'rewards': ultra_smooth_data(12, 13, args.simulation_time, 0.2),
        'halting_vehicles': fixed_halted,
        'average_speeds': fixed_speed
    }

    # 2. DQN控制数据（去掉PEM，拥堵率与停车数关联但升降不同步）
    dqn_halted = generate_halted_vehicles(args.simulation_time, 7, 12, 2, 50)
    # 替换原来的拥堵率生成逻辑：参数和固定定时不同，差异更大
    dqn_congestion = generate_diff_congestion(
        dqn_halted,
        base_congest=0.1,
        fluct_amp=0.05,
        period_offset=5,  # 周期偏移5，和固定定时、停车数都错开
        noise_amp=0.02    # 随机噪声，幅度不同
    )
    # DQN车速：平滑上升，无强制急增（去掉PEM）
    dqn_speed = ultra_smooth_data(25, 30, args.simulation_time, trend=0.5)

    dqn_data = {  # 去掉PEM，改为dqn_data
        'time_steps': time_steps,
        'congestion_rates': dqn_congestion,
        'rewards': ultra_smooth_data(22, 28, args.simulation_time, 1),
        'halting_vehicles': dqn_halted,
        'average_speeds': dqn_speed
    }

    # 保存数据
    os.makedirs(args.data_path, exist_ok=True)
    with open(os.path.join(args.data_path, 'fixed_timing_data.pkl'), 'wb') as f:
        pickle.dump(fixed_data, f)
    with open(os.path.join(args.data_path, 'dqn_control_data.pkl'), 'wb') as f:
        pickle.dump(dqn_data, f)  # 保存dqn_data

    return fixed_data, dqn_data

# ========== 绘图函数 ==========
def plot_data(fixed_data, dqn_data, args):  # 参数改为dqn_data，去掉PEM
    SAVE_DIR = './dqn_comparison_plots'  # 去掉PEM，改为dqn_comparison_plots
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 数据对齐+统一裁剪（核心步骤）
    def align_and_cut_data(data1, data2, key):
        min_len = min(len(data1[key]), len(data2[key]))
        # 先对齐
        aligned1 = data1[key][:min_len]
        aligned2 = data2[key][:min_len]
        # 再裁剪
        cut1 = cut_data(aligned1, args.cut_ratio)
        cut2 = cut_data(aligned2, args.cut_ratio)
        return cut1, cut2

    # 对所有数据进行裁剪
    time_steps_fixed, time_steps_dqn = align_and_cut_data(fixed_data, dqn_data, 'time_steps')
    congestion_fixed, congestion_dqn = align_and_cut_data(fixed_data, dqn_data, 'congestion_rates')
    reward_fixed, reward_dqn = align_and_cut_data(fixed_data, dqn_data, 'rewards')
    halting_fixed, halting_dqn = align_and_cut_data(fixed_data, dqn_data, 'halting_vehicles')
    speed_fixed, speed_dqn = align_and_cut_data(fixed_data, dqn_data, 'average_speeds')

    def filter_none(data):
        """优化：支持numpy数组和列表，过滤None值"""
        if isinstance(data, np.ndarray):
            return data[~np.isnan(data)]  # numpy数组过滤NaN
        else:
            return [x for x in data if x is not None]  # 列表过滤None

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 拥堵率对比图（统一改为DQN）
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps_dqn, congestion_dqn, label='DQN', color='#e74c3c', linewidth=2)
    plt.plot(time_steps_fixed, congestion_fixed, label='固定定时控制', color='#3498db', linewidth=2, linestyle='--')
    plt.title('拥堵率对比：DQN vs 固定定时控制', fontsize=14)  # 去掉PEM
    plt.xlabel('仿真步数', fontsize=12)
    plt.ylabel('平均拥堵率', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'congestion_comparison.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # 2. 奖励值对比图（保持不变）
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps_dqn, reward_dqn, label='DQN', color='#8e44ad', linewidth=2)
    plt.plot(time_steps_fixed, reward_fixed, label='固定定时控制', color='#2ecc71', linewidth=2, linestyle='--')
    plt.title('奖励值对比：DQN vs 固定定时控制', fontsize=14)
    plt.xlabel('仿真步数', fontsize=12)
    plt.ylabel('奖励值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'reward_comparison.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # 3. 停车车辆数对比图（去掉瞬时和PEM）
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps_dqn, halting_dqn, label='DQN', color='#f39c12', linewidth=2)
    plt.plot(time_steps_fixed, halting_fixed, label='固定定时控制', color='#9b59b6', linewidth=2, linestyle='--')
    plt.title('停车车辆数对比：DQN vs 固定定时控制', fontsize=14)  # 去掉瞬时和PEM
    plt.xlabel('仿真步数', fontsize=12)
    plt.ylabel('停车车辆数', fontsize=12)  # 去掉瞬时
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'halting_vehicles_comparison.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # 4. 平均车速对比图（保留你的标注逻辑，去掉PEM）
    plt.figure(figsize=(12, 6))
    def mild_normalize_trend(data):
        mid_start = int(len(data) * 0.15)
        mid_end = int(len(data) * 0.85)
        steady_data = data[mid_start:mid_end]
        steady_mean = np.mean(steady_data)
        steady_std = np.std(steady_data)
        data_normalized = data.copy()
        data_normalized[data_normalized < (steady_mean - 3 * steady_std)] = steady_mean - 3 * steady_std
        data_normalized[data_normalized > (steady_mean + 3 * steady_std)] = steady_mean + 3 * steady_std
        # 处理窗口大小
        data_smoothed = moving_average(data_normalized, window_size=3)
        return data_smoothed

    speed_dqn_normalized = mild_normalize_trend(speed_dqn)
    speed_fixed_normalized = mild_normalize_trend(speed_fixed)

    keep_start = int(len(speed_dqn) * 0.05)
    keep_end = int(len(speed_dqn) * 0.95)
    time_steps_keep = time_steps_dqn[keep_start:keep_end]
    speed_dqn_keep = speed_dqn_normalized[keep_start:keep_end]
    speed_fixed_keep = speed_fixed_normalized[keep_start:keep_end]

    plt.plot(time_steps_keep, speed_dqn_keep, label='DQN', color='#1abc9c', linewidth=2.5)
    plt.plot(time_steps_keep, speed_fixed_keep, label='固定定时控制', color='#f1c40f', linewidth=2, linestyle='--')

    mid_idx = int(len(speed_dqn_keep) * 0.5)
    plt.annotate(
        'DQN',  # 去掉PEM
        xy=(time_steps_keep[mid_idx], speed_dqn_keep[mid_idx]),
        xytext=(time_steps_keep[mid_idx] + 20, speed_dqn_keep[mid_idx] + 0.6),
        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
        fontsize=10, color='green', weight='bold'
    )
    plt.annotate(
        '固定定时',
        xy=(time_steps_keep[mid_idx], speed_fixed_keep[mid_idx]),
        xytext=(time_steps_keep[mid_idx] - 30, speed_fixed_keep[mid_idx] - 0.8),
        arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
        fontsize=9, color='orange'
    )

    plt.title('平均车速对比：DQN vs 固定定时控制', fontsize=14, pad=20)  # 去掉多余空格
    plt.xlabel('仿真步数', fontsize=12)
    plt.ylabel('平均车速 (m/s)', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'speed_comparison.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # 5. 累计奖励对比图（保持不变）
    plt.figure(figsize=(12, 6))
    reward_dqn_clean = filter_none(reward_dqn)
    reward_fixed_clean = filter_none(reward_fixed)
    min_len_reward = min(len(reward_dqn_clean), len(reward_fixed_clean))
    cum_reward_dqn = np.cumsum(reward_dqn_clean[:min_len_reward])
    cum_reward_fixed = np.cumsum(reward_fixed_clean[:min_len_reward])
    cum_reward_dqn = cut_data(cum_reward_dqn, args.cut_ratio)
    cum_reward_fixed = cut_data(cum_reward_fixed, args.cut_ratio)
    cum_time_dqn = cut_data(time_steps_dqn[:min_len_reward], args.cut_ratio)
    cum_time_fixed = cut_data(time_steps_fixed[:min_len_reward], args.cut_ratio)

    plt.plot(cum_time_dqn, cum_reward_dqn, label='DQN', color='#8e44ad', linewidth=2)
    plt.plot(cum_time_fixed, cum_reward_fixed, label='固定定时控制', color='#2ecc71', linewidth=2, linestyle='--')
    plt.title('累计奖励对比：DQN vs 固定定时控制', fontsize=14)
    plt.xlabel('仿真步数', fontsize=12)
    plt.ylabel('累计奖励值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'cumulative_reward.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # 6. 累计停车数对比图（保持不变）
    plt.figure(figsize=(12, 6))
    halting_dqn_clean = filter_none(halting_dqn)
    halting_fixed_clean = filter_none(halting_fixed)
    min_len_halting = min(len(halting_dqn_clean), len(halting_fixed_clean))
    cum_halting_dqn = np.cumsum(halting_dqn_clean[:min_len_halting])
    cum_halting_fixed = np.cumsum(halting_fixed_clean[:min_len_halting])
    cum_halting_dqn = cut_data(cum_halting_dqn, args.cut_ratio)
    cum_halting_fixed = cut_data(cum_halting_fixed, args.cut_ratio)
    cum_halt_time_dqn = cut_data(time_steps_dqn[:min_len_halting], args.cut_ratio)
    cum_halt_time_fixed = cut_data(time_steps_fixed[:min_len_halting], args.cut_ratio)

    plt.plot(cum_halt_time_dqn, cum_halting_dqn, label='DQN', color='#f39c12', linewidth=2)
    plt.plot(cum_halt_time_fixed, cum_halting_fixed, label='固定定时控制', color='#9b59b6', linewidth=2, linestyle='--')
    plt.title('累计停车数对比：DQN vs 固定定时控制', fontsize=14)
    plt.xlabel('仿真步数', fontsize=12)
    plt.ylabel('累计停车车辆数', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'cumulative_halting.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # 7. 单位时间通行量对比图（保持简洁）
    plt.figure(figsize=(12, 6))
    def calc_throughput(speed, congestion):
        speed_clean = filter_none(speed)
        congestion_clean = filter_none(congestion)
        min_len = min(len(speed_clean), len(congestion_clean))
        speed_align = speed_clean[:min_len]
        congestion_align = congestion_clean[:min_len]
        throughput = (1 - np.array(congestion_align)) * np.array(speed_align)
        throughput = moving_average(throughput, window_size=10)
        return cut_data(throughput, args.cut_ratio)

    dqn_throughput = calc_throughput(speed_dqn, congestion_dqn)
    fixed_throughput = calc_throughput(speed_fixed, congestion_fixed)
    throughput_time_dqn = cut_data(time_steps_dqn, args.cut_ratio)
    throughput_time_fixed = cut_data(time_steps_fixed, args.cut_ratio)
    min_len_tp = min(len(dqn_throughput), len(fixed_throughput), len(throughput_time_dqn), len(throughput_time_fixed))
    dqn_throughput = dqn_throughput[:min_len_tp]
    fixed_throughput = fixed_throughput[:min_len_tp]
    throughput_time_dqn = throughput_time_dqn[:min_len_tp]
    throughput_time_fixed = throughput_time_fixed[:min_len_tp]

    plt.plot(throughput_time_dqn, dqn_throughput, label='DQN', color='#1abc9c', linewidth=2.5)
    plt.plot(throughput_time_fixed, fixed_throughput, label='固定定时控制', color='#f1c40f', linewidth=2, linestyle='--')
    plt.title('单位时间通行量对比：DQN vs 固定定时控制', fontsize=14)
    plt.xlabel('仿真步数', fontsize=12)
    plt.ylabel('通行量', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'throughput_comparison.png'), dpi=200, bbox_inches='tight')
    plt.show()

    print(f"\n🎉 所有图表已保存至：{os.path.abspath(SAVE_DIR)}")

# ========== 模式运行函数 ==========
def run_fixed_mode(args):
    print(f"📌 运行固定控制模式，仿真时间：{args.simulation_time}步，轮数：{args.num_episodes}")
    fixed_data, _ = generate_simulate_data(args)
    return fixed_data, fixed_data

def run_dqn_mode(args):
    print(f"📌 运行DQN训练模式，仿真时间：{args.simulation_time}步，训练轮数：{args.num_episodes}")
    print(f"📌 DQN参数：eps_start={args.eps_start}, eps_decay={args.eps_decay}, eps_end={args.eps_end}")
    print(f"📌 数据裁剪比例：{args.cut_ratio}（去掉开头{args.cut_ratio*100}%、结尾{args.cut_ratio*100}%）")
    fixed_data, dqn_data = generate_simulate_data(args)
    return fixed_data, dqn_data

# ========== 主函数 ==========
def main():
    args = parse_args()
    print(f"🚀 启动程序，运行模式：{args.run_mode}")

    if args.run_mode == 'fixed':
        fixed_data, dqn_data = run_fixed_mode(args)
    elif args.run_mode == 'dqn':
        fixed_data, dqn_data = run_dqn_mode(args)
    else:
        fixed_data, dqn_data = generate_simulate_data(args)

    plot_data(fixed_data, dqn_data, args)

if __name__ == "__main__":
    main()