import argparse

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import traci
import sumolib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAnalyzer:
    def __init__(self, net_file, route_file, simulation_time, use_gui=False,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 learning_rate=0.0002, batch_size=256, memory_size=20000):  # 调整学习率
        self.net_file = net_file
        self.route_file = route_file
        self.simulation_time = simulation_time
        self.use_gui = use_gui
        self.sumoBinary = 'sumo-gui' if use_gui else 'sumo'
        self.traci = traci
        self.lanes_id = []
        self.lanes_length = {}
        self.vehicle_size_min_gap = 7.5
        self.congestion_rates = []
        self.lane_densities = []
        self.directional_congestion_rates = {}
        self.traffic_signal_phases = []
        self.time_steps = []
        self.rewards = []

        # DQN 参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # 获取状态和动作维度
        self.start_simulation()
        self.input_dim = len(self.get_lanes_density())
        self.output_dim = len(self.traci.trafficlight.getAllProgramLogics(self.traci.trafficlight.getIDList()[0])[0].phases)
        self.close_simulation()

        # 初始化 DQN 网络
        self.model = DQN(self.input_dim, self.output_dim)
        self.target_model = DQN(self.input_dim, self.output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def start_simulation(self):
        sumo_cmd = [sumolib.checkBinary(self.sumoBinary), '-n', self.net_file, '-r', self.route_file]
        self.traci.start(sumo_cmd)
        self.lanes_id = self.traci.lane.getIDList()
        self.lanes_length = {lane_id: self.traci.lane.getLength(lane_id) for lane_id in self.lanes_id}

    def collect_data(self):
        self.start_simulation()
        for step in range(int(self.simulation_time)):
            state = np.array(self.get_lanes_density())
            action = self.act(state)
            self.traci.trafficlight.setPhase(self.traci.trafficlight.getIDList()[0], action)
            self.traci.simulationStep()
            self.time_steps.append(step)

            # 收集车道密度
            density = self.get_lanes_density()
            self.lane_densities.append(density)

            # 收集停车车辆数
            halting_vehicles = [self.traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.lanes_id]

            # 收集平均速度
            average_speeds = [self.traci.lane.getLastStepMeanSpeed(lane_id) for lane_id in self.lanes_id]

            # 收集交通信号灯相位
            traffic_signal_id = self.traci.trafficlight.getIDList()[0]
            current_phase = self.traci.trafficlight.getPhase(traffic_signal_id)
            self.traffic_signal_phases.append(current_phase)

            # 计算拥堵率
            congestion_rate = self.compute_congestion_rate()
            self.congestion_rates.append(congestion_rate)

            # 计算按方向划分的拥堵情况
            self.compute_directional_congestion()

            # 改进奖励函数，更注重减少停车车辆和拥堵率
            reward = -sum(halting_vehicles) * 2 + sum(average_speeds) * 0.2 - congestion_rate * 100  # 加大拥堵率惩罚力度
            self.rewards.append(reward)

            next_state = np.array(self.get_lanes_density())
            self.remember(state, action, reward, next_state)
            self.replay()

        self.close_simulation()

    def get_lanes_density(self):
        return [min(1, self.traci.lane.getLastStepVehicleNumber(lane_id) / (self.lanes_length[lane_id] / self.vehicle_size_min_gap))
                for lane_id in self.lanes_id]

    def compute_congestion_rate(self):
        total_congestion_rate = 0
        for lane_id in self.lanes_id:
            halting_vehicles = self.traci.lane.getLastStepHaltingNumber(lane_id)
            max_vehicles = self.lanes_length[lane_id] / self.vehicle_size_min_gap
            congestion_rate = halting_vehicles / max_vehicles if max_vehicles > 0 else 0
            total_congestion_rate += congestion_rate
        average_congestion_rate = total_congestion_rate / len(self.lanes_id)
        return average_congestion_rate

    def compute_directional_congestion(self):
        # 假设车道 ID 有一定的命名规则，例如 'n_t_0' 表示北方向
        north_lanes = [lane for lane in self.lanes_id if lane.startswith('n_')]
        south_lanes = [lane for lane in self.lanes_id if lane.startswith('s_')]
        east_lanes = [lane for lane in self.lanes_id if lane.startswith('e_')]
        west_lanes = [lane for lane in self.lanes_id if lane.startswith('w_')]

        def compute_direction_congestion(lanes):
            total_congestion = 0
            for lane in lanes:
                halting_vehicles = self.traci.lane.getLastStepHaltingNumber(lane)
                max_vehicles = self.lanes_length[lane] / self.vehicle_size_min_gap
                congestion_rate = halting_vehicles / max_vehicles if max_vehicles > 0 else 0
                total_congestion += congestion_rate
            return total_congestion / len(lanes) if lanes else 0

        north_congestion = compute_direction_congestion(north_lanes)
        south_congestion = compute_direction_congestion(south_lanes)
        east_congestion = compute_direction_congestion(east_lanes)
        west_congestion = compute_direction_congestion(west_lanes)

        directions = ['North', 'South', 'East', 'West']
        congestion = [north_congestion, south_congestion, east_congestion, west_congestion]

        for i, direction in enumerate(directions):
            if direction not in self.directional_congestion_rates:
                self.directional_congestion_rates[direction] = []
            self.directional_congestion_rates[direction].append(congestion[i])

    def visualize_data(self):
        # 绘制拥堵率随时间变化的图表
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_steps, self.congestion_rates)
        plt.title('Congestion Rate over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Congestion Rate')
        plt.grid(True)
        plt.savefig('congestion_rate.png')
        plt.show()

        # 绘制车道密度热图
        lane_densities_array = np.array(self.lane_densities)
        plt.figure(figsize=(10, 6))
        sns.heatmap(lane_densities_array.T, cmap='viridis', cbar_kws={'label': 'Lane Density'})
        plt.title('Lane Density Heatmap')
        plt.xlabel('Time Step')
        plt.ylabel('Lane ID')
        plt.savefig('lane_density_heatmap.png')
        plt.show()

        # 绘制按方向划分的拥堵情况对比图
        plt.figure(figsize=(10, 6))
        for direction, congestion in self.directional_congestion_rates.items():
            plt.plot(self.time_steps, congestion, label=direction)
        plt.title('Directional Congestion Rate over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Congestion Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig('directional_congestion.png')
        plt.show()

        # 绘制奖励随时间变化的图表
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_steps, self.rewards)
        plt.title('Reward over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig('reward_over_time.png')
        plt.show()

    def close_simulation(self):
        self.traci.close()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        action = torch.argmax(q_values, dim=1).item()

        # 双 DQN 策略
        if np.random.rand() < 0.5:
            target_q_values = self.target_model(state)
            action = torch.argmax(target_q_values, dim=1).item()

        return action

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        target_q_values = q_values.clone()
        target_q_values[range(self.batch_size), actions] = rewards + self.gamma * max_next_q_values

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Traffic Analyzer using SUMO')
    # 1. 路网文件路径（改为你的绝对路径）
    parser.add_argument('--net-file', type=str,
                        default='C:/Users/60322/Desktop/PEM实践/nets/2way-single-intersection/single-intersection.net.xml',
                        help='Path to the SUMO network file')
    # 2. 车流文件路径（改为你的PEMS车流文件）
    parser.add_argument('--route-file', type=str,
                        default='C:/Users/60322/Desktop/PEM实践/nets/2way-single-intersection/intersection.rou.xml',
                        help='Path to the SUMO route file')
    # 3. 仿真时长（改为10000秒，和你之前的配置对齐）
    parser.add_argument('--simulation-time', type=float, default=10000,
                        help='Simulation time in seconds')
    parser.add_argument('--use-gui', action='store_true',
                        help='Use SUMO GUI for visualization')
    args = parser.parse_args()

    analyzer = DQNAnalyzer(args.net_file, args.route_file, args.simulation_time, args.use_gui)
    analyzer.collect_data()
    analyzer.visualize_data()