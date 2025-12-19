from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
import math
import os
import pickle
import traci
import sumolib
import numpy as np
#运行 SUMO 交通仿真，分别用固定定时控制和DQN 强化学习控制交通灯，采集交通数据用于对比
# 导入车流生成函数
from sumo_video_traffic_generator import generate_traffic_from_video

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备是: {device}")

# 命令行参数定义（复用你的配置）
FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 50, 'time range for random skip')
flags.DEFINE_float('simulation_time', 100, 'simulation duration (建议测试用100秒)')
flags.DEFINE_integer('yellow_time', 2, 'yellow phase time')
flags.DEFINE_integer('delta_rs_update_time', 10, 'reward calc interval')
flags.DEFINE_string('net_file',
                    'C:\\Users\\60322\\Desktop\\视频实践\\nets\\2way-single-intersection\\single-intersection.net.xml',
                    'SUMO net file')
flags.DEFINE_string('route_file',
                    'C:\\Users\\60322\\Desktop\\视频实践\\nets\\2way-single-intersection\\video_based_route.rou.xml',
                    'SUMO route file')
flags.DEFINE_bool('use_gui', True, 'use sumo-gui')
flags.DEFINE_integer('num_episodes', 10, 'DQN training episodes (测试用10)')
flags.DEFINE_string('mode', 'train', 'train/test')
flags.DEFINE_float('eps_start', 0.8, 'initial epsilon')
flags.DEFINE_float('eps_end', 0.05, 'final epsilon')
flags.DEFINE_integer('eps_decay', 80000, 'epsilon decay')
flags.DEFINE_integer('target_update', 1000, 'target net update')
flags.DEFINE_float('gamma', 0.95, 'discount factor')
flags.DEFINE_integer('batch_size', 32, 'batch size')
# 新增：预训练网络文件路径（解决AttributeError）
flags.DEFINE_string('network_file', '', '预训练网络文件路径')
# 新增：选择运行模式（fixed/dqn）
flags.DEFINE_string('run_mode', 'dqn', '运行模式：fixed(定时控制)/dqn(DQN控制)')

# ========== 1. 定时控制数据采集 ==========
def collect_fixed_timing_data():
    """运行固定定时控制，采集数据"""
    print("\n" + "=" * 60)
    print("📌 开始采集定时控制数据...")

    # 生成车流文件
    old_route = FLAGS.route_file
    if os.path.exists(old_route):
        os.remove(old_route)
    generated_route = generate_traffic_from_video()

    # 初始化SUMO
    sumo_binary = sumolib.checkBinary('sumo-gui' if FLAGS.use_gui else 'sumo')
    sumo_cmd = [
        sumo_binary, '-n', FLAGS.net_file, '-r', generated_route,
        '--start', '--quit-on-end', '--no-warnings'
    ]
    traci.start(sumo_cmd)

    # 初始化数据存储
    lanes_id = traci.lane.getIDList()
    lanes_length = {l: traci.lane.getLength(l) for l in lanes_id}
    vehicle_size_min_gap = 7.5

    fixed_data = {
        'time_steps': [],
        'congestion_rates': [],
        'rewards': [],
        'halting_vehicles': [],
        'average_speeds': []
    }

    # 固定配时逻辑（南北20s/东西20s/黄灯2s）
    tl_id = traci.trafficlight.getIDList()[0]
    phase_durations = [20, 2, 20, 2]
    phase_index = 0
    phase_step = 0

    # 运行仿真
    for step in range(int(FLAGS.simulation_time)):
        traci.simulationStep()

        # 切换相位
        phase_step += 1
        if phase_step >= phase_durations[phase_index]:
            phase_index = (phase_index + 1) % 4
            phase_step = 0
            traci.trafficlight.setPhase(tl_id, phase_index)

        # 采集拥堵率
        total_congestion = 0
        halting_veh = []
        avg_speeds = []
        for l in lanes_id:
            halt = traci.lane.getLastStepHaltingNumber(l)
            speed = traci.lane.getLastStepMeanSpeed(l)
            max_veh = lanes_length[l] / vehicle_size_min_gap
            total_congestion += halt / max_veh if max_veh > 0 else 0
            halting_veh.append(halt)
            avg_speeds.append(speed)

        # 计算奖励（和DQN一致）
        congestion_rate = total_congestion / len(lanes_id)
        reward = -sum(halting_veh) * 3 + sum(avg_speeds) * 0.3 - congestion_rate * 150
        # 保存数据
        fixed_data['time_steps'].append(step)
        fixed_data['congestion_rates'].append(congestion_rate)
        fixed_data['rewards'].append(reward)
        fixed_data['halting_vehicles'].append(sum(halting_veh))
        fixed_data['average_speeds'].append(np.mean(avg_speeds))

    # 保存数据
    traci.close()
    with open('./fixed_timing_data.pkl', 'wb') as f:
        pickle.dump(fixed_data, f)
    print("✅ 定时控制数据已保存：fixed_timing_data.pkl")
    print("=" * 60)


# ========== 2. DQN控制数据采集（基于你的main函数） ==========
def collect_dqn_data():
    """运行DQN控制，采集数据"""
    print("\n" + "=" * 60)
    print("📌 开始采集DQN控制数据...")

    # 生成车流文件
    old_route_file = FLAGS.route_file
    if os.path.exists(old_route_file):
        os.remove(old_route_file)
    try:
        generated_route_path = generate_traffic_from_video()
    except Exception as e:
        print(f"❌ 车流生成失败：{e}")
        return

    # 初始化环境
    env = SumoEnv(
        net_file=FLAGS.net_file,
        route_file=generated_route_path,
        skip_range=FLAGS.skip_range,
        simulation_time=FLAGS.simulation_time,
        yellow_time=FLAGS.yellow_time,
        delta_rs_update_time=FLAGS.delta_rs_update_time,
        use_gui=FLAGS.use_gui
    )

    # 初始化DQN代理
    replay_buffer = ReplayBuffer(capacity=20000)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DqnAgent(
        FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma,
        FLAGS.eps_start, FLAGS.eps_end, FLAGS.eps_decay,
        input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file
    )

    # 数据存储
    dqn_data = {
        'time_steps': [],
        'congestion_rates': [],
        'rewards': [],
        'halting_vehicles': [],
        'average_speeds': []
    }
    global_step = 0

    # 训练循环
    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False

        while not done:
            state = env.compute_state
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            next_state, reward, done, info = env.step(action)

            # 采集数据（仅有效动作）
            if info['do_action'] is not None:
                # 核心修复：直接通过traci获取车道ID，不依赖env的lanes_id属性
                lanes_id = traci.lane.getIDList()
                halting_veh = [traci.lane.getLastStepHaltingNumber(l) for l in lanes_id]
                avg_speeds = [traci.lane.getLastStepMeanSpeed(l) for l in lanes_id]
                lanes_length = {l: traci.lane.getLength(l) for l in lanes_id}
                vehicle_size_min_gap = 7.5

                # 计算拥堵率
                total_congestion = 0
                for l in lanes_id:
                    halt = traci.lane.getLastStepHaltingNumber(l)
                    max_veh = lanes_length[l] / vehicle_size_min_gap
                    total_congestion += halt / max_veh if max_veh > 0 else 0
                congestion_rate = total_congestion / len(lanes_id)

                # 保存数据
                dqn_data['time_steps'].append(global_step)
                dqn_data['congestion_rates'].append(congestion_rate)
                dqn_data['rewards'].append(reward)
                dqn_data['halting_vehicles'].append(sum(halting_veh))
                dqn_data['average_speeds'].append(np.mean(avg_speeds))

                global_step += 1

            # 无效动作处理
            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False

            # DQN学习
            replay_buffer.add(env.train_state, next_state, reward, info['do_action'])
            agent.learn()

        env.close()
        print(f"✅ Episode {episode} 完成 | 总步数: {global_step}")

    # 保存DQN数据
    with open('./dqn_control_data.pkl', 'wb') as f:
        pickle.dump(dqn_data, f)
    print("✅ DQN控制数据已保存：dqn_control_data.pkl")
    print("=" * 60)


# ========== 主函数 ==========
def main(argv):
    del argv

    # 根据run_mode选择运行方式
    if FLAGS.run_mode == 'fixed':
        collect_fixed_timing_data()
    elif FLAGS.run_mode == 'dqn':
        collect_dqn_data()
    else:
        print("❌ 无效的run_mode！可选：fixed/dqn")


if __name__ == '__main__':
    app.run(main)