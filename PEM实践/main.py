from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备是: {device}")

FLAGS = flags.FLAGS
# ===================== 核心修改：替换为你的文件路径 =====================
# 1. 随机跳过时间
flags.DEFINE_integer('skip_range', 50, 'time(seconds) range for skip randomly at the beginning')
# 2. 仿真时间（和你SUMO配置的10000秒一致）
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
# 3. 黄灯时间（和路网文件的2秒一致）
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
# 4. 奖励计算间隔
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')

# 5. 路网文件路径（你的实际路径）
flags.DEFINE_string('net_file',
                    'C:/Users/60322/Desktop/PEM实践/nets/2way-single-intersection/single-intersection.net.xml', '')
# 6. 车流文件路径（你的PEMS流量文件）
flags.DEFINE_string('route_file',
                    'C:/Users/60322/Desktop/PEM实践/nets/2way-single-intersection/auto_generated.rou.xml', '')

# 7. 使用图形界面（启动SUMO-GUI）
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
# 8. 训练回合数
flags.DEFINE_integer('num_episodes', 301, '')
# 9. 网络类型
flags.DEFINE_string('network', 'dqn', '')
# 10. 训练/测试模式
flags.DEFINE_string('mode', 'train', '')
# 11. epsilon参数
flags.DEFINE_float('eps_start', 1.0, '')
flags.DEFINE_float('eps_end', 0.1, '')
flags.DEFINE_integer('eps_decay', 83000, '')
# 12. 目标网络更新频率
flags.DEFINE_integer('target_update', 3000, '')
# 13. 预训练网络路径（训练时留空）
flags.DEFINE_string('network_file', '', '')
# 14. 折扣因子
flags.DEFINE_float('gamma', 0.95, '')
# 15. 批量大小
flags.DEFINE_integer('batch_size', 32, '')

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"


# 主函数
def main(argv):
    del argv
    # 创建SUMO环境（加载你的路网+PEMS车流）
    env = SumoEnv(net_file=FLAGS.net_file,
                  route_file=FLAGS.route_file,
                  skip_range=FLAGS.skip_range,
                  simulation_time=FLAGS.simulation_time,
                  yellow_time=FLAGS.yellow_time,
                  delta_rs_update_time=FLAGS.delta_rs_update_time,
                  use_gui=FLAGS.use_gui
                  )
    replay_buffer = ReplayBuffer(capacity=20000)

    # 获取状态/动作维度
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    # 创建DQN代理
    agent = DqnAgent(FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma, FLAGS.eps_start, FLAGS.eps_end,
                     FLAGS.eps_decay, input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file)

    # 训练循环
    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()  # 重置SUMO环境，启动新回合
        env.train_state = initial_state
        done = False
        invalid_action = False
        while not done:
            state = env.compute_state
            # 代理选择信号灯动作
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            # 执行动作，更新信号灯并获取奖励
            next_state, reward, done, info = env.step(action)

            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False

            # 存储经验到回放缓冲区
            replay_buffer.add(env.train_state, next_state, reward, info['do_action'])
            # DQN学习优化信号灯策略
            agent.learn()

        # 关闭当前回合的SUMO环境
        env.close()

        # 打印训练日志
        print('i_episode:', episode)
        print('eps_threshold = :', FLAGS.eps_end + (FLAGS.eps_start - FLAGS.eps_end) *
              math.exp(-1. * replay_buffer.steps_done / FLAGS.eps_decay))
        print('learn_steps:', agent.learn_steps)


if __name__ == '__main__':
    app.run(main)