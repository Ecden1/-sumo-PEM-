import gym
import traci
import sumolib
import random

from environment.traffic_signal import TrafficSignal


class SumoEnv(gym.Env):
    def __init__(
        self,
        net_file: str,
        route_file: str,
        skip_range: int,
        simulation_time: float,
        yellow_time: int,
        delta_rs_update_time: int,
        use_gui: bool = False
    ):
        # 保存网络文件路径
        self._net = net_file
        # 保存路由文件路径
        self._route = route_file
        # 保存随机跳过的时间范围
        self.skip_range = skip_range
        # 保存仿真时间
        self.simulation_time = simulation_time
        # 保存是否使用图形界面的标志
        self.use_gui = use_gui
        # 保存黄灯时间
        self.yellow_time = yellow_time
        # 初始化训练状态
        self.train_state = None
        # 初始化上一阶段状态
        self.last_phase_state = None
        # 初始化动作改变时间
        self.change_action_time = None
        # 初始化SUMO连接
        self.sumo = None
        # 初始化SUMO二进制文件路径
        self.sumoBinary = 'sumo'
        # 如果使用图形界面，则修改SUMO二进制文件路径
        if self.use_gui:
            self.sumoBinary = 'sumo-gui'

        # 启动SUMO仿真，仅用于获取交通信号ID
        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        # 获取SUMO连接
        conn = traci
        # 获取交通信号ID
        self.ts_id = traci.trafficlight.getIDList()[0]
        # 创建TrafficSignal实例
        self.traffic_signal = TrafficSignal(ts_id=self.ts_id,
                                            yellow_time=self.yellow_time,
                                            simulation_time=simulation_time,
                                            delta_rs_update_time=delta_rs_update_time,
                                            sumo=conn)
        # 关闭SUMO连接
        conn.close()

    def step(self, action):
        # 初始化下一状态
        next_state = None
        # 初始化奖励
        reward = None
        # 初始化回合结束标志
        done = False
        # 初始化信息字典
        info = {'do_action': None}
        # 初始化奖励计算标志
        start = False

        # 改变交通信号相位
        do_action = self.traffic_signal.change_phase(action)
        # 如果动作无效，则直接返回
        if do_action is None:
            return next_state, reward, done, info

        # 执行一步仿真
        self.sumo.simulationStep()

        # 如果处于黄灯阶段且动作改变时间未设置，则设置动作改变时间
        if do_action == -1 and self.change_action_time is None:
            self.change_action_time = self.sumo.simulation.getTime() + self.yellow_time

        # 如果动作改变时间已到，则重置动作改变时间并计算训练状态
        if self.change_action_time is not None and self.sumo.simulation.getTime() >= self.change_action_time:
            self.change_action_time = None
            self.train_state = self._compute_state()
            start = True

        # 计算下一状态
        next_state = self._compute_next_state()
        # 计算奖励
        reward = self._compute_reward(start, do_action)
        # 计算回合结束标志
        done = self._compute_done()
        # 更新信息字典
        info = {'do_action': do_action}
        return next_state, reward, done, info

    def _random_skip(self, skip_range=30):
        # 设置TrafficSignal的SUMO连接
        self.traffic_signal.sumo = self.sumo
        # 随机选择一个绿色相位
        rand_idx = random.randint(0, len(self.traffic_signal.all_green_phases)-1)
        # 重置黄灯相位
        self.traffic_signal.yellow_phase = None
        # 设置当前绿色相位
        self.traffic_signal.green_phase = self.traffic_signal.all_green_phases[rand_idx]
        # 设置交通信号的状态
        self.sumo.trafficlight.setRedYellowGreenState(self.traffic_signal.ts_id, self.traffic_signal.green_phase.state)
        # 更新结束时间
        self.traffic_signal.update_end_time()
        # 重置奖励状态更新时间
        self.traffic_signal.rs_update_time = 0

        # 随机选择跳过的秒数
        skip_seconds = random.randint(0, skip_range)
        # 计算初始状态
        initial_state = self._compute_state()
        # 执行跳过操作
        for s in range(skip_seconds):
            # 随机选择一个动作
            rand_idx = random.randint(0, len(self.traffic_signal.all_green_phases)-1)
            # 执行动作并获取下一状态
            next_state, _, _, _ = self.step(rand_idx)
            # 如果下一状态有效，则更新初始状态
            if next_state is not None:
                initial_state = next_state

        return initial_state

    def reset(self):
        # 构建SUMO启动命令
        sumo_cmd = [sumolib.checkBinary(self.sumoBinary), '-n', self._net, '-r', self._route,
                    '--time-to-teleport', '1000']
        # 如果使用图形界面，则添加额外的参数
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
        # 启动SUMO仿真
        traci.start(sumo_cmd)
        # 获取SUMO连接
        self.sumo = traci

        # 如果使用图形界面，则设置视图模式
        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            # 重置随机跳过的时间范围
            self.skip_range = 0

        # 执行随机跳过操作并返回初始状态
        return self._random_skip(self.skip_range)

    def close(self):
        # 关闭SUMO连接
        self.sumo.close()

    def render(self):
        # 渲染方法，暂时为空
        pass

    # 计算当前状态
    def _compute_state(self):
        return self.traffic_signal.compute_state()

    # 获取当前状态的属性方法
    @property
    def compute_state(self):
        return self._compute_state()

    # 计算下一状态
    def _compute_next_state(self):
        next_state = self.traffic_signal.compute_next_state()
        return next_state

    # 计算奖励
    def _compute_reward(self, start, do_action):
        ts_reward = self.traffic_signal.compute_reward(start, do_action)
        return ts_reward

    # 计算回合结束标志
    def _compute_done(self):
        # 获取当前仿真时间
        current_time = self.sumo.simulation.getTime()
        # 如果当前时间超过仿真时间，则回合结束
        if current_time > self.simulation_time:
            done = True
        else:
            done = False

        return done

    # 获取观测空间
    @property
    def observation_space(self):
        return self.traffic_signal.observation_space

    # 获取动作空间
    @property
    def action_space(self):
        return self.traffic_signal.action_space