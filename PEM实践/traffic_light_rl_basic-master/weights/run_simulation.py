import traci
import pandas as pd
from pathlib import Path

# 1. 配置路径
CSV_PATH = Path(r"C:\Users\60322\Desktop\Single_Sensor_Time_Series.csv")  # 你的CSV路径
SUMO_NET_PATH = Path(r"C:\Users\60322\Desktop\traffic_light_rl_basic-master\nets\2way-single-intersection\single-intersection.net.xml") # 你的SUMO路网文件路径
SUMO_ROU_PATH = Path(r"C:\Users\60322\Desktop\single-intersection-gen.rou.xml")  # 上面修改的车流文件

# 2. 加载CSV数据（提取时间、flow、speed）
df = pd.read_csv(CSV_PATH)
# 转换时间标签为秒（CSV时间步：5分钟=300秒/步）
df["time_second"] = [t * 300 for t in range(len(df))]
sensor_id = df["sensor_id"].iloc[0]  # 获取当前传感器ID（如Sensor_23）
sumo_edge_id = "t_n"  # 传感器对应SUMO路段ID（需与你的路网匹配）

# 3. 启动SUMO仿真
sumo_cmd = [
    "sumo-gui",  # 带GUI界面，方便查看；无GUI用"sumo"
    "-n", str(SUMO_NET_PATH),
    "-r", str(SUMO_ROU_PATH),
    "--step-length", "1",  # 仿真步长1秒
    "--duration", str(df["time_second"].max() + 300)  # 仿真时长覆盖CSV所有时间步
]
traci.start(sumo_cmd)

# 4. 实时更新路段速度（匹配CSV的speed）
for idx, row in df.iterrows():
    current_time = row["time_second"]
    target_speed = row["speed"]  # CSV当前时间步的真实速度

    # 推进仿真到当前时间步
    traci.simulationStep(current_time)

    # 更新对应路段的最大速度（强制车辆按真实速度行驶）
    traci.edge.setMaxSpeed(sumo_edge_id, target_speed)

    # 打印日志（可选）
    print(f"时间：{current_time}秒 | 路段：{sumo_edge_id} | 真实速度：{target_speed:.1f} km/h")

# 5. 结束仿真
traci.close()
print("仿真完成！")