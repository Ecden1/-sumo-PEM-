import numpy as np
from gym import spaces


class TrafficSignal:
    def __init__(
            self,
            ts_id: str,
            yellow_time: int,
            simulation_time: float,
            delta_rs_update_time: int,
            sumo
    ):
        # 保存交通信号ID
        self.ts_id = ts_id
        # 保存黄灯时间
        self.yellow_time = yellow_time
        # 保存仿真时间
        self.simulation_time = simulation_time
        # 保存奖励状态更新时间间隔
        self.delta_rs_update_time = delta_rs_update_time
        # 初始化奖励状态更新时间
        self.rs_update_time = 0

        self.sumo = sumo
        # 初始化绿色相位
        self.green_phase = None
        # 初始化黄色相位
        self.yellow_phase = None
        # 初始化结束时间
        self.end_time = 0
        # 获取所有相位
        self.all_phases = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        # 筛选出所有绿色相位
        self.all_green_phases = [phase for phase in self.all_phases if 'y' not in phase.state]
        # 获取交通信号控制的车道ID
        self.lanes_id = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        # 计算每条车道的长度
        self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes_id}
        # 定义观测空间
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.lanes_id), dtype=np.float32),
            high=np.ones(len(self.lanes_id), dtype=np.float32))
        # 定义动作空间
        self.action_space = spaces.Discrete(len(self.all_green_phases))
        # 初始化上一次的测量值，用于计算奖励
        self.last_measure = 0
        # 初始化持续奖励标志
        self.continue_reward = False
        # 初始化车道车辆字典
        self.dict_lane_veh = None
        # 新增：拥堵率相关变量
        self.congestion_threshold = 0.7  # 拥堵密度阈值，可根据需要调整
        self.congestion_history = []  # 存储每个时间步的拥堵率
        self.current_congestion_rate = 0  # 当前拥堵率

    def change_phase(self, new_green_phase):
        """
        :param new_green_phase: 新的绿色相位索引
        :return: do_action -> 实际执行的动作；如果为None，表示新的绿色相位不合适，需要重新选择
        """
        # 将动作索引转换为相位对象
        new_green_phase = self.all_green_phases[new_green_phase]
        # 获取当前仿真时间
        current_time = self.sumo.simulation.getTime()
        # 如果处于黄灯阶段
        if self.yellow_phase is not None:
            # 如果黄灯阶段已结束
            if current_time >= self.end_time:
                # 重置黄灯相位
                self.yellow_phase = None
                # 更新结束时间
                self.update_end_time()
                # 设置交通信号的状态为绿色相位
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                # 实际执行的动作是绿色相位
                do_action = self.green_phase
            else:
                # 实际执行的动作是黄灯相位
                do_action = self.yellow_phase
        else:
            # 如果旧的绿色相位已结束
            if current_time >= self.end_time:
                # 如果新的绿色相位和当前绿色相位相同
                if new_green_phase.state == self.green_phase.state:
                    # 当前相位已达到最大运行时间，需要选择其他绿色相位
                    do_action = None
                else:
                    # 计算拥堵率
                    congestion_rate = self._compute_congestion_rate()
                    # 根据拥堵率动态调整绿色相位时长
                    if congestion_rate > 0.8:
                        new_green_phase.duration = 60  # 严重拥堵时增加绿色相位时长
                    elif congestion_rate > 0.5:
                        new_green_phase.duration = 40  # 中度拥堵时增加绿色相位时长
                    else:
                        new_green_phase.duration = 33  # 轻度拥堵时保持默认时长

                    # 计算黄灯状态
                    yellow_state = ''
                    for s in range(len(new_green_phase.state)):
                        if self.green_phase.state[s] == 'G' and new_green_phase.state[s] == 'r':
                            yellow_state += 'y'
                        else:
                            yellow_state += self.green_phase.state[s]
                    # 创建黄灯相位对象
                    self.yellow_phase = self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                    # 设置交通信号的状态为黄灯相位
                    self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase.state)
                    # 更新当前绿色相位
                    self.green_phase = new_green_phase
                    # 更新奖励状态更新时间
                    self.rs_update_time = current_time + self.yellow_time + self.delta_rs_update_time
                    # 更新结束时间
                    self.update_end_time()
                    # 实际执行的动作是黄灯相位
                    do_action = self.yellow_phase
            else:
                # 实际执行的动作是当前绿色相位
                do_action = self.green_phase

        # 如果动作无效，则返回None
        if do_action is None:
            return None

        # 将相位对象转换为动作索引
        if 'y' in do_action.state:
            do_action = -1
        else:
            for i, green_phase in enumerate(self.all_green_phases):
                if do_action.state == green_phase.state:
                    do_action = i
                    break

        # 计算拥堵率
        self._compute_congestion_rate()

        return do_action

    def update_end_time(self):
        # 获取当前仿真时间
        current_time = self.sumo.simulation.getTime()
        # 如果不是黄灯阶段
        if self.yellow_phase is None:
            # 结束时间为当前时间加上绿色相位的持续时间
            self.end_time = current_time + self.green_phase.duration
        else:
            # 结束时间为当前时间加上黄灯时间
            self.end_time = current_time + self.yellow_time

    def compute_reward(self, start, do_action):
        # 初始化奖励更新标志
        update_reward = False
        # 获取当前仿真时间
        current_time = self.sumo.simulation.getTime()
        # 如果当前时间超过奖励状态更新时间
        if current_time >= self.rs_update_time:
            # 设置奖励状态更新时间为不可达
            self.rs_update_time = self.simulation_time + self.delta_rs_update_time
            # 设置奖励更新标志为True
            update_reward = True

        # 调用内部方法计算奖励
        return self._choose_min_waiting_time(start, update_reward, do_action)

    def _choose_min_waiting_time(self, start, update_reward, do_action):
        # 如果开始计算奖励
        if start:
            # 初始化车道车辆字典
            self.dict_lane_veh = {}
            # 遍历所有车道
            for lane_id in self.lanes_id:
                # 获取每条车道上停止的车辆数量
                self.dict_lane_veh[lane_id] = self.sumo.lane.getLastStepHaltingNumber(lane_id)
            # 合并等待车辆数量
            dict_action_wait_num = [self.dict_lane_veh['n_t_0'] + self.dict_lane_veh['s_t_0'],
                                    self.dict_lane_veh['n_t_1'] + self.dict_lane_veh['s_t_1'],
                                    self.dict_lane_veh['e_t_0'] + self.dict_lane_veh['w_t_0'],
                                    self.dict_lane_veh['e_t_1'] + self.dict_lane_veh['w_t_1']]
            # 选择等待车辆数量最多的动作
            best_action = np.argmax(dict_action_wait_num)
            # 如果选择的动作是最优动作
            if best_action == do_action:
                # 上一次的测量值为1
                self.last_measure = 1
            else:
                # 上一次的测量值为-1
                self.last_measure = -1

        # 如果需要更新奖励
        if update_reward:
            return self.last_measure
        else:
            return None

    def compute_next_state(self):
        # 获取当前仿真时间
        current_time = self.sumo.simulation.getTime()
        # 如果当前时间超过奖励状态更新时间
        if current_time >= self.rs_update_time:
            # 计算车道密度
            density = self.get_lanes_density()
            # 将车道密度转换为numpy数组
            next_state = np.array(density, dtype=np.float32)
            return next_state
        else:
            return None

    def compute_state(self):
        # 计算车道密度
        density = self.get_lanes_density()
        # 将车道密度转换为numpy数组
        state = np.array(density, dtype=np.float32)
        return state

    def get_lanes_density(self):
        # 车辆最小间距
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        # 计算每条车道的密度
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane_id) / (
                    self.lanes_length[lane_id] / vehicle_size_min_gap))
                for lane_id in self.lanes_id]

    def _compute_congestion_rate(self):
        """计算当前拥堵率"""
        # 获取各车道密度
        densities = self.get_lanes_density()

        # 计算拥堵车道数量
        congested_lanes = sum(1 for density in densities if density >= self.congestion_threshold)

        # 计算拥堵率
        self.current_congestion_rate = congested_lanes / len(densities)

        # 记录历史拥堵率
        current_time = self.sumo.simulation.getTime()
        self.congestion_history.append((current_time, self.current_congestion_rate))

        return self.current_congestion_rate

    def get_current_congestion_rate(self):
        """获取当前拥堵率"""
        return self.current_congestion_rate

    def get_average_congestion_rate(self):
        """获取平均拥堵率"""
        if not self.congestion_history:
            return 0
        return sum(rate for _, rate in self.congestion_history) / len(self.congestion_history)

    def get_congestion_history(self):
        """获取拥堵率历史记录"""
        return self.congestion_history