from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
import math
import os  # 新增：用于删除旧文件
# ========== 新增：导入车流生成脚本的核心函数 ==========
from sumo_video_traffic_generator import generate_traffic_from_video  # 直接导入生成函数

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备是: {device}")

# 使用absl库定义命令行参数
FLAGS = flags.FLAGS
# 定义随机跳过的时间范围（秒），用于在开始时随机跳过一段时间
flags.DEFINE_integer('skip_range', 50, 'time(seconds) range for skip randomly at the beginning')
# 定义仿真时间
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
# 定义黄灯时间
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
# 定义计算奖励的时间间隔
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
# 定义网络文件路径（改为你的实际路径）
flags.DEFINE_string('net_file', 'C:\\Users\\60322\\Desktop\\视频实践\\nets\\2way-single-intersection\\single-intersection.net.xml', '')
# 定义路由文件路径（默认值，会被动态生成的文件替换）
flags.DEFINE_string('route_file', 'C:\\Users\\60322\\Desktop\\视频实践\\nets\\2way-single-intersection\\video_based_route.rou.xml', '')
# 定义是否使用图形界面
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
# 定义训练的回合数
flags.DEFINE_integer('num_episodes', 301, '')
# 定义使用的网络类型
flags.DEFINE_string('network', 'dqn', '')
# 定义模式，如训练或测试
flags.DEFINE_string('mode', 'train', '')
# 定义epsilon-greedy策略的初始值
flags.DEFINE_float('eps_start', 1.0, '')
# 定义epsilon-greedy策略的最终值
flags.DEFINE_float('eps_end', 0.1, '')
# 定义epsilon-greedy策略的衰减步数
flags.DEFINE_integer('eps_decay', 83000, '')
# 定义目标网络更新的频率
flags.DEFINE_integer('target_update', 3000, '')
# 定义网络文件路径
flags.DEFINE_string('network_file', '', '')
# 定义折扣因子
flags.DEFINE_float('gamma', 0.95, '')
# 定义批量大小
flags.DEFINE_integer('batch_size', 32, '')

# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 主函数
def main(argv):
    # 忽略命令行参数列表中的第一个元素（脚本名称）
    del argv

    # ========== 核心修复：真正执行车流生成函数 ==========
    print("="*50)
    print("🚀 开始生成基于视频的SUMO车流文件...")
    # 第一步：删除旧的车流文件（避免缓存/占用）
    old_route_file = FLAGS.route_file  # 从命令行参数获取路径，避免硬编码
    if os.path.exists(old_route_file):
        os.remove(old_route_file)
        print(f"🗑️ 已删除旧车流文件：{old_route_file}")
    # 第二步：调用核心函数生成新的车流文件（返回生成的文件路径）
    try:
        generated_route_path = generate_traffic_from_video()  # 真正执行生成！
        print(f"✅ 车流文件生成完成，路径：{generated_route_path}")
    except Exception as e:
        print(f"❌ 车流文件生成失败：{e}")
        return  # 生成失败则退出，避免后续报错
    print("="*50)

    # ========== 确认使用生成的路由文件 ==========
    env = SumoEnv(net_file=FLAGS.net_file,
                  route_file=generated_route_path,  # 使用动态生成的路径（和FLAGS.route_file一致）
                  skip_range=FLAGS.skip_range,
                  simulation_time=FLAGS.simulation_time,
                  yellow_time=FLAGS.yellow_time,
                  delta_rs_update_time=FLAGS.delta_rs_update_time,
                  use_gui=FLAGS.use_gui
                  )
    # 创建经验回放缓冲区实例
    replay_buffer = ReplayBuffer(capacity=20000)

    # 获取环境的观测空间维度
    input_dim = env.observation_space.shape[0]
    # 获取环境的动作空间维度
    output_dim = env.action_space.n
    # 创建DqnAgent代理实例（完全保留你原有无报错的写法）
    agent = DqnAgent(FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma, FLAGS.eps_start, FLAGS.eps_end,
                     FLAGS.eps_decay, input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file)

    # 开始训练回合
    for episode in range(FLAGS.num_episodes):
        # 重置环境并获取初始状态
        initial_state = env.reset()
        # 设置训练状态为初始状态
        env.train_state = initial_state
        # 初始化回合结束标志
        done = False
        # 初始化无效动作标志
        invalid_action = False
        # 当回合未结束时，持续进行交互
        while not done:
            # 获取当前状态
            state = env.compute_state
            # 代理选择动作
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            # 执行动作并获取下一个状态、奖励、回合结束标志和其他信息
            next_state, reward, done, info = env.step(action)
            # 如果动作无效，则跳过本次循环
            if info['do_action'] is None:
                invalid_action = True
                continue
            # 重置无效动作标志
            invalid_action = False

            # 将经验添加到回放缓冲区
            replay_buffer.add(env.train_state, next_state, reward, info['do_action'])
            # 代理进行学习
            agent.learn()

        # 关闭环境
        env.close()

        # 打印当前回合数
        print('i_episode:', episode)
        # 打印当前的epsilon值
        print('eps_threshold = :', FLAGS.eps_end + (FLAGS.eps_start - FLAGS.eps_end) *
              math.exp(-1. * replay_buffer.steps_done / FLAGS.eps_decay))
        # 打印学习步数
        print('learn_steps:', agent.learn_steps)

# 程序入口
if __name__ == '__main__':
    app.run(main)