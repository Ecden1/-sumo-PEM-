import cv2
import pandas as pd
from datetime import datetime
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import random
import csv
import time

#根据PEM的CSV文件导入生成车流文件
class TrafficDataFetcher:
    def __init__(self, pems_csv_paths):
        # 替换视频/CSV为PEMS多方向CSV路径
        self.pems_csv_paths = pems_csv_paths  # {"east":path, "south":path, "west":path, "north":path}
        self.last_flow_rates = {"east": 0.0, "west": 0.0, "north": 0.0, "south": 0.0}

    def get_pems_avg_flow(self):
        """读取PEMS CSV，返回各方向每小时平均流量"""
        dir_data = {}
        print(f"[{datetime.now()}] 开始读取PEMS数据...")
        for direction, csv_path in self.pems_csv_paths.items():
            try:
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"PEMS文件不存在：{csv_path}")

                df = pd.read_csv(csv_path)
                if "flow_5min" not in df.columns:
                    raise ValueError(f"{csv_path}缺少'flow_5min'列")

                # 计算每小时平均流量（flow_5min是5分钟流量，×12转为每小时）
                avg_flow = max(df["flow_5min"].mean() * 12, 20)  # 保底20辆/小时
                dir_data[direction] = int(avg_flow)
                print(f"[{datetime.now()}] {direction}方向PEMS平均流量：{avg_flow:.0f}辆/小时")
            except Exception as e:
                print(f"[{datetime.now()}] {direction}方向PEMS读取失败：{str(e)}")
                dir_data[direction] = 20  # 保底值

        return dir_data

    def calculate_flow_rate(self):
        """兼容原有接口，返回各方向流量（辆/秒）"""
        dir_flow_hour = self.get_pems_avg_flow()
        flow_rates = {}

        for direction, flow_hour in dir_flow_hour.items():
            flow_rates[direction] = flow_hour / 3600  # 转为辆/秒
            self.last_flow_rates[direction] = flow_rates[direction]
            print(f"[{datetime.now()}] {direction}方向车流速度：{flow_rates[direction]:.4f} 辆/秒")

        # 兼容原有Leaving/Entering逻辑，取均值
        avg_flow_per_second = sum(flow_rates.values()) / len(flow_rates)
        self.last_flow_rates["Leaving"] = avg_flow_per_second
        self.last_flow_rates["Entering"] = avg_flow_per_second

        return self.last_flow_rates


def generate_sumo_route_file(
        base_vehs_per_hour,
        lane_ratios,
        output_route_file="video_based_route.rou.xml",
        output_csv_file="deepsort_template.csv",
        sim_duration=400000,
        depart_speed="max",
        depart_pos="base",
        generate_csv=True
):
    # 完全保留你原有代码，仅修改流量分配逻辑
    target_dir = os.path.dirname(output_route_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"📁 自动创建目录：{target_dir}")

    # 定义所有路由（和你原始代码完全一致）
    all_routes = [
        {"id": "route_ew", "edges": "e_t t_w"}, {"id": "route_en", "edges": "e_t t_n"},
        {"id": "route_es", "edges": "e_t t_s"},
        {"id": "route_we", "edges": "w_t t_e"}, {"id": "route_wn", "edges": "w_t t_n"},
        {"id": "route_ws", "edges": "w_t t_s"},
        {"id": "route_ns", "edges": "n_t t_s"}, {"id": "route_nw", "edges": "n_t t_w"},
        {"id": "route_ne", "edges": "n_t t_e"},
        {"id": "route_sn", "edges": "s_t t_n"}, {"id": "route_se", "edges": "s_t t_e"},
        {"id": "route_sw", "edges": "s_t t_w"}
    ]
    lane_route_map = {
        "east": ["route_ew", "route_en", "route_es"],
        "west": ["route_we", "route_wn", "route_ws"],
        "north": ["route_ns", "route_nw", "route_ne"],
        "south": ["route_sn", "route_se", "route_sw"]
    }
    ne_sw_turn_routes = ["route_en", "route_ne", "route_sw", "route_ws"]
    route_ids = [r["id"] for r in all_routes]

    # 转向比例（保留你改好的配置，基于PEMS分配）
    ratio_config = {
        "route_ew": 0.5, "route_en": 0.2, "route_es": 0.3,
        "route_we": 0.6, "route_wn": 0.2, "route_ws": 0.2,
        "route_ns": 0.6, "route_nw": 0.2, "route_ne": 0.2,
        "route_sn": 0.5, "route_se": 0.3, "route_sw": 0.2
    }

    # 流量计算（仅改保底数值：50→100，80→150，基于PEMS基数分配）
    flow_config = {}
    for lane, ratio in lane_ratios.items():
        lane_total_vehs = round(base_vehs_per_hour * ratio)
        lane_total_vehs = max(lane_total_vehs, 100 if lane in ["north", "south"] else 150)
        for route_id in lane_route_map[lane]:
            flow_config[route_id] = round(lane_total_vehs * ratio_config[route_id])

    # 生成路由文件（和你原始代码完全一致）
    root = ET.Element("routes")
    for route in all_routes:
        route_elem = ET.SubElement(root, "route")
        route_elem.set("id", route["id"])
        route_elem.set("edges", route["edges"])
    for route_id in route_ids:
        flow_elem = ET.SubElement(root, "flow")
        flow_elem.set("id", f"flow_{route_id}_video")
        flow_elem.set("route", route_id)
        flow_elem.set("begin", "0")
        flow_elem.set("end", str(sim_duration))
        flow_elem.set("vehsPerHour", str(flow_config[route_id]))
        flow_elem.set("departSpeed", depart_speed)
        flow_elem.set("departPos", depart_pos)
        flow_elem.set("departLane", "0")  # 保持和你一致

    # 格式化XML（和你原始一致）
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    with open(output_route_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    # 生成CSV（和你原始一致）
    if generate_csv:
        with open(output_csv_file, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["vehicle_id", "depart_time", "start_edge", "end_edge", "vtype"])
            writer.writeheader()
            veh_id = 1
            for t in range(0, 300, 1):
                for route_id in route_ids:
                    veh_per_second = flow_config[route_id] / 3600
                    if random.random() < veh_per_second:
                        parts = route_id.split('_')
                        dir_part = parts[1]
                        dir1 = dir_part[0]
                        dir2 = dir_part[1]
                        start_edge = f"{dir1}_t"
                        end_edge = f"t_{dir2}"
                        writer.writerow({
                            "vehicle_id": f"deepsort_{veh_id}",
                            "depart_time": t,
                            "start_edge": start_edge,
                            "end_edge": end_edge,
                            "vtype": "car"
                        })
                        veh_id += 1

    # 日志（保留你的逻辑，仅适配PEMS数据）
    print("=" * 80)
    print(f"🚨 测试模式：转向比例保留，车流适度增加！")
    print(f"✅ 车流文件生成完成！")
    print(f"📄 SUMO路由文件：{output_route_file}")
    print(f"📄 DeepSORT CSV模板：{output_csv_file}")
    print(f"📊 基准车流基数（PEMS均值）：{base_vehs_per_hour} 辆/小时")
    print(f"📋 测试配置：转向比例保留，车流适度增加！")
    print("📈 各进口道车流详情（总流量/直行/东北西南拐弯/占比）：")
    total_cross_veh = 0
    total_ne_sw_turn = 0
    for lane, ratio in lane_ratios.items():
        lane_total = sum([flow_config[rid] for rid in lane_route_map[lane]])
        total_cross_veh += lane_total
        straight_flow = sum([flow_config[rid] for rid in lane_route_map[lane] if
                             rid in ["route_ew", "route_we", "route_ns", "route_sn"]])
        ne_sw_flow = sum([flow_config[rid] for rid in lane_route_map[lane] if rid in ne_sw_turn_routes])
        total_ne_sw_turn += ne_sw_flow
        print(f"\n  🚦 {lane}口（总流量：{lane_total}辆/小时）：")
        print(f"    - 直行：{straight_flow}辆/小时（占{straight_flow / lane_total * 100:.0f}%）")
        print(f"    - 东北/西南拐弯：{ne_sw_flow}辆/小时（占{ne_sw_flow / lane_total * 100:.0f}%）")
        print(f"    - 其他拐弯：{lane_total - straight_flow - ne_sw_flow}辆/小时")

    print(f"\n📊 全局统计：")
    print(f"  - 十字路口总车流：{total_cross_veh} 辆/小时")
    print(f"  - 东北/西南向拐弯总流量：{total_ne_sw_turn}辆/小时（占全局{total_ne_sw_turn / total_cross_veh * 100:.0f}%）")
    print(
        f"  - 直行总流量：{sum([flow_config[rid] for rid in ['route_ew', 'route_we', 'route_ns', 'route_sn']])}辆/小时（占全局{sum([flow_config[rid] for rid in ['route_ew', 'route_we', 'route_ns', 'route_sn']]) / total_cross_veh * 100:.0f}%）")
    print("=" * 80)

    return flow_config, output_route_file


def read_deepsort_data_safe(deepsort_csv_path, retry=3, sleep_time=0.1):
    # 完全保留你原有代码
    vehicles_to_add = []
    added_vehicles = set()
    while retry > 0:
        try:
            with open(deepsort_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    veh_id = row['vehicle_id']
                    if veh_id not in added_vehicles:
                        vehicles_to_add.append({
                            "id": veh_id,
                            "depart_time": float(row['depart_time']),
                            "start_edge": row['start_edge'],
                            "end_edge": row['end_edge'],
                            "vtype": "car"
                        })
                        added_vehicles.add(veh_id)
            break
        except PermissionError:
            time.sleep(sleep_time)
            retry -= 1
        except Exception as e:
            print(f"读取DeepSORT数据失败：{e}")
            retry -= 1
    return vehicles_to_add


# 核心调用函数（替换为PEMS读取）
# 核心调用函数（替换为PEMS读取）
def generate_traffic_from_pems():
    # 替换为你的PEMS CSV路径（修正Path格式+字符串转义）
    PEMS_CSV_PATHS = {
        "east": Path(r"C:\Users\60322\Desktop\PEM实践\PEM\PEMS08_sensor_8_data.csv"),
        "south": Path(r"C:\Users\60322\Desktop\PEM实践\PEM\PEMS08_sensor_84_data.csv"),
        "west": Path(r"C:\Users\60322\Desktop\PEM实践\PEM\PEMS08_sensor_124_data.csv"),
        "north": Path(r"C:\Users\60322\Desktop\PEM实践\PEM\PEMS08_sensor_162_data.csv")
    }
    # 修正输出路径的字符串转义（用r前缀避免转义）
    OUTPUT_ROUTE_FILE = r"C:\Users\60322\Desktop\PEM实践\nets\2way-single-intersection\auto_generated.rou.xml"
    OUTPUT_CSV_FILE = r"C:\Users\60322\Desktop\PEM实践\nets\2way-single-intersection\deepsort_template.csv"

    # 读取PEMS数据
    fetcher = TrafficDataFetcher(PEMS_CSV_PATHS)
    flow_rates = fetcher.calculate_flow_rate()

    # 计算基数（兼容原有逻辑，取均值×3600转为每小时）
    avg_flow_per_second = (flow_rates["Leaving"] + flow_rates["Entering"]) / 2
    base_vehs_per_hour = round(avg_flow_per_second * 3600)

    # 改基数：保底300，上限400（保留你的逻辑）
    if base_vehs_per_hour <= 0:
        base_vehs_per_hour = 300
    base_vehs_per_hour = min(base_vehs_per_hour, 400)
    print(f"⚠️ 测试用车流基数：{base_vehs_per_hour} 辆/小时（适度增流）")

    # 改方向比例：保留你的配置
    lane_ratios = {
        "east": 1.2,
        "west": 1.0,
        "north": 0.6,
        "south": 0.5
    }

    # 生成文件（完全保留你的逻辑）
    flow_config, route_file_path = generate_sumo_route_file(
        base_vehs_per_hour=base_vehs_per_hour,
        lane_ratios=lane_ratios,
        output_route_file=OUTPUT_ROUTE_FILE,
        output_csv_file=OUTPUT_CSV_FILE
    )

    return route_file_path

# 独立运行入口
if __name__ == "__main__":
    # 替换为PEMS入口
    generate_traffic_from_pems()