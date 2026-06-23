import socket
import threading
# import pygame
import cv2
import numpy as np
import gradio as gr

import io
import base64

from time import sleep, time
from threading import Thread
from PIL import Image
from drone import TelloPy
from drone import TelloPyServer
from prompt import get_message_template
from utils import extract_code_blocks
from rich import print
from openai import OpenAI

import requests
import random

import re
import math

import os
import sys

# ====== v3 MODIFIED: Initialize Log Directory and Helper Function ======
LOG_DIR = 'log'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

CLIENT1_LOG_FILE = os.path.join(LOG_DIR, 'client1.txt')
CLIENT2_LOG_FILE = os.path.join(LOG_DIR, 'client2.txt')

# ====== 自动保存 Server 控制台日志到 server_console.txt ======
class ServerLogger(object):
    def __init__(self, filename=os.path.join(LOG_DIR, 'server_console.txt')):
        self.terminal = sys.stdout
        # 每次重启 Server 都会覆盖之前的日志
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    # ====== 修复 uvicorn/gradio 的 isatty 报错 ======
    def isatty(self):
        if hasattr(self.terminal, 'isatty'):
            return self.terminal.isatty()
        return False
    # ======================================================

# 接管标准输出
sys.stdout = ServerLogger()
print("\n" + "="*50)
print("Server Started - Console Output will be logged to log/server_console.txt")
print("="*50 + "\n")
# ====================================================================

def append_client_log(drone_id: int, step: int, text: str):
    """Append LLM output to the respective client log file."""
    filename = CLIENT1_LOG_FILE if drone_id == 1 else CLIENT2_LOG_FILE
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"### Step {step}\n{text}\n\n")
    except Exception as e:
        print(f"Failed to write log for drone {drone_id}: {e}")
# ====================================================================

SERVER_IP = '10.113.163.114'
SERVER_PORT = 9000
server_sock = None
first_frame_flag = True
drone_1 : TelloPyServer | None = None
drone_2 : TelloPyServer | None = None

clients = []
# ser
client = OpenAI(
    api_key='EMPTY',
    base_url='http://10.113.182.9:8001/v1',
)

answer=[
"""
```python
drone.move_left(35)
```

Explanation: Find a drone in the image in the left part of the image and move left to keep it in the center of the image.

Description: The image shows an indoor setting with a drone in the left part of the image. The drone has four propellers. The lighting is bright, and the image is clear. I will move left to keep the drone in the center of the image.
""",
"""
```python
drone.move_right(30)
```

Explanation: Find a drone in the image in the right part of the image. According to the user's instruction, I will move right to keep it in the center of the image.

Description: The image shows an indoor setting with a drone in the right part of the image, likely in a office or laboratory environment. I will move right to keep the drone in the center of the image.
""",
"""
```python
drone.move_up(30)
```

Explanation: A drone is in the top part of the image. Move up to keep it in the center of the image.
Description: The image shows a drone in the top part of the image. The drone is flying in the air. I will move up to keep the drone in the center of the image.
""",
"""
```python
drone.move_down(30)
```

Explanation: Searching for the drone in the image. The drone is in the bottom part of the image. Move down to keep it in the center of the image.

Description: The image shows a drone in the bottom part of the image. The drone is flying in the air. Move the camera down to keep the drone in the center of the image.
"""
]


current_action=0

# --- 多机协作全局状态板 ---
# ====== 修改：增加 base_x 和 base_y 作为相对起飞原点 ======
drone_states = {
    1: { # 对应 drone_1
        "base_x": 0.0, "base_y": 0.0,             # 物理基准坐标
        "x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0, # 实时更新的物理坐标
        "landmarks": set(),                       # 已经搜索过的地标集合
        "intent": ""
    },
    2: { # 对应 drone_2
        "base_x": 1.0, "base_y": 0.0,             # 假设起飞时 drone_2 在 drone_1 右侧 1.0 米处
        "x": 1.0, "y": 0.0, "z": 1.5, "yaw": 0.0, 
        "landmarks": set(),
        "intent": ""
    }
}

# ====== 修改点 1：定义全局变量（用于全异步控制） ======
current_instruction = None  # 存储当前网页下发的全局任务指令（如 'rack'）
is_drone1_stop = True       # 1号机的任务运行状态（初始为停止，下发任务时激活）
is_drone2_stop = True       # 2号机的任务运行状态

step_drone1 = 1
step_drone2 = 1
# =======================================================

# ====== 完全交给物理反馈，不再通过大模型动作更新 Yaw ======
def update_virtual_state(drone_id: int, llm_output: str):
    """
    提取地标和意图。物理坐标(X, Y, Z, Yaw)已完全由底层MVO和IMU接管，大模型不再推算！
    """
    state = drone_states[drone_id]
    
    # 1. 提取地标
    landmark_match = re.search(r'\[Landmark\]:\s*(.+)', llm_output, re.IGNORECASE)
    if landmark_match:
        items = landmark_match.group(1).split(',')
        for item in items:
            cleaned_item = item.strip().lower()
            if cleaned_item and cleaned_item != 'none':
                state["landmarks"].add(cleaned_item)
                
    # 2. 提取意图
    intent_match = re.search(r'Intent:\s*(.+)', llm_output, re.IGNORECASE)
    if intent_match:
        state["intent"] = intent_match.group(1).strip()
        
    print(f"[Drone {drone_id} True State] Pos:(X:{state['x']:.2f}m, Y:{state['y']:.2f}m, Z:{state['z']:.2f}m, Yaw:{state['yaw']:.1f}°), Landmarks: {list(state['landmarks'])}, Intent: {state['intent']}")


def ask(messages):
    start_time = time()

    response = client.chat.completions.create(
        model='Qwen2.5-VL-32B-Instruct',
        # model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        messages=messages,
        # response_format={"type": "json_object"},
    )
    print(response.choices[0].message.content)
    output_text = response.choices[0].message.content

    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": output_text,
                }
            ]
        }
    )

    end_time = time()

    print(f"Time taken: {end_time - start_time:.2f}s")
    return output_text, messages


messages = None
drone1_messages = None
drone2_messages = None


def get_last_assistant_message():
    global messages, drone1_messages, drone2_messages
    if messages is None:
        return 'No command yet'
    res = []
    for message in messages[::-1]:
        if message['role'] == 'assistant':
            res.append(message['content'][0]['text'])
    if len(res) > 0:
        for i in range(len(res)):
            res[i] = f'### Step {i + 1}\n' + res[i]

        return "\n\n".join(res[::-1])

    return 'No command yet'


def get_drone_messages(drone_id):
    global drone1_messages, drone2_messages
    # Select the correct message source
    messages = drone1_messages if drone_id == 1 else drone2_messages

    if not messages:
        return 'No command yet'

    res = []
    # Logic to extract assistant text from your specific message format
    for message in messages:
        if message.get('role') == 'assistant':
            content = message['content']
            # Handle if content is a list of dicts (like in your snippet) or just a string
            text = content[0]['text'] if isinstance(content, list) else content
            res.append(text)

    if res:
        # Format: Step 1: text, Step 2: text...
        formatted = [f'### Step {i + 1}\n{msg}' for i, msg in enumerate(res)]
        return "\n\n".join(formatted)

    return 'No command yet'

def delete_image(messages):
    for message in messages:
        if message['content'][0]['type'] == 'image' or message['content'][0]['type'] == 'image_url':
            message['content'].pop(0)
            if message['role'] == 'system':
                message['content'][0]['text'] = "Continue to generate the next step."
    return messages


def process_instruction(instruction):
    global current_instruction, is_drone1_stop, is_drone2_stop
    global drone1_messages, drone2_messages, step_drone1, step_drone2

    print(f"New global task received: {instruction}")
    current_instruction = instruction

    try:
        with open(CLIENT1_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"========== NEW INSTRUCTION: {instruction} ==========\n\n")
        with open(CLIENT2_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"========== NEW INSTRUCTION: {instruction} ==========\n\n")
    except Exception as e:
        print(f"Failed to write separator to log: {e}")

    # 动态唤醒
    if drone_1 is not None:
        drone1_messages = get_message_template()
        step_drone1 = 1
        is_drone1_stop = False
        
    if drone_2 is not None:
        drone2_messages = get_message_template()
        step_drone2 = 1
        is_drone2_stop = False


# ====== 修改点 2 ======
def send_command(drone: TelloPyServer, output_text: str):
    if drone is None or drone.socket is None:
        return
    try:
        drone.socket.sendall(output_text.encode())
    except ConnectionResetError:
        print("Connection reset by peer")
    except BrokenPipeError:
        print("Broken pipe")
    except Exception as e:
        print(f"{e}")


# 修改为支持热插拔与单机显示的动态视频流
def video():
    cv2.namedWindow("Merged Drone Cameras", cv2.WINDOW_NORMAL)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # 如果两台无人机都没连上，就休眠等待
        if drone_1 is None and drone_2 is None:
            sleep(1)
            continue
            
        try:
            # 获取1号机画面，如果没连上或画面没准备好，用黑色背景占位
            if drone_1 is not None and drone_1.current_image is not None:
                img_1 = drone_1.current_image
            else:
                img_1 = np.zeros((720, 960, 3), dtype=np.uint8)
                cv2.putText(img_1, "Waiting for Drone 1...", (280, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 获取2号机画面，如果没连上或画面没准备好，用黑色背景占位
            if drone_2 is not None and drone_2.current_image is not None:
                img_2 = drone_2.current_image
            else:
                img_2 = np.zeros((720, 960, 3), dtype=np.uint8)
                cv2.putText(img_2, "Waiting for Drone 2...", (280, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 左右拼接
            img_merged = np.hstack((img_1, img_2))
            img_resize_merged = cv2.resize(img_merged, (0, 0), fx=0.8, fy=0.8)
            
            # 显示合并画面
            cv2.imshow("Merged Drone Cameras", img_resize_merged)
            
        except Exception as e:
            print(f"Error in video loop: {e}")

            sleep(0.1)

    cv2.destroyAllWindows()


def drone1_worker_loop():
    global drone_1, drone1_messages, is_drone1_stop, current_instruction, step_drone1
    is_first_message = True
    
    while True:
        if drone_1 is None or current_instruction is None or is_drone1_stop:
            sleep(1)
            is_first_message = True
            continue
            
        if drone_1.current_image is None:
            sleep(0.5)
            continue
            
        try:
            last_image_drone1 = Image.fromarray(cv2.cvtColor(drone_1.current_image, cv2.COLOR_BGR2RGB))
            buf_drone1 = io.BytesIO()
            last_image_drone1.save(buf_drone1, format='JPEG')
            last_image_str_drone1 = base64.b64encode(buf_drone1.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Drone 1 Image Conversion Error: {e}")
            sleep(0.5)
            continue

        state_1 = drone_states[1]
        state_2 = drone_states[2]
        
        # 修复了使用 .copy() 防止多线程报错
        info_2 = (
            f"\n\n[MY STATUS] Drone 1 is at relative pos (X:{state_1['x']:.2f}, Y:{state_1['y']:.2f}, Z:{state_1['z']:.2f}, Yaw:{state_1['yaw']:.1f}°)."
            f"\n[TEAMMATE STATUS] Drone 2 is at relative pos (X:{state_2['x']:.2f}, Y:{state_2['y']:.2f}, Z:{state_2['z']:.2f}, Yaw:{state_2['yaw']:.1f}°). Landmarks found: {list(state_2['landmarks'].copy())}."
            f"\n[TEAMMATE INTENT]: {state_2['intent']}"
        )

        # 5步动态唤醒机制(Drone 1)
        if step_drone1 % 5 == 0:
            info_2 += f"\n\n[SYSTEM REMINDER]: You have searched for {step_drone1} steps. If you still have not found the target '{current_instruction}', you might be stuck in a blind spot. Please immediately execute a large-angle rotation (e.g., rotate_cw(90) or rotate_ccw(120)) to explore entirely new areas!"
        
        
        drone1_messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{last_image_str_drone1}"}},
                {"type": "text", "text": current_instruction + info_2}
            ]
        })

        output_text_drone1, drone1_messages = ask(drone1_messages)
        print('DRONE 1 Assistant:', output_text_drone1)

        append_client_log(1, step_drone1, output_text_drone1)
        step_drone1 += 1
        update_virtual_state(1, output_text_drone1)
        send_command(drone_1, output_text_drone1)
        
        for text in ['break', 'drone.land()']:
            if text in output_text_drone1:
                is_drone1_stop = True
                
        drone1_messages = delete_image(drone1_messages)
        
        if is_first_message:
            sleep(1)
            is_first_message = False

def drone2_worker_loop():
    global drone_2, drone2_messages, is_drone2_stop, current_instruction, step_drone2
    is_first_message = True
    
    while True:
        if drone_2 is None or current_instruction is None or is_drone2_stop:
            sleep(1)
            is_first_message = True
            continue
            
        if drone_2.current_image is None:
            sleep(0.5)
            continue
            
        try:
            last_image_drone2 = Image.fromarray(cv2.cvtColor(drone_2.current_image, cv2.COLOR_BGR2RGB))
            buf_drone2 = io.BytesIO()
            last_image_drone2.save(buf_drone2, format='JPEG')
            last_image_str_drone2 = base64.b64encode(buf_drone2.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Drone 2 Image Conversion Error: {e}")
            sleep(0.5)
            continue

        state_1 = drone_states[1]
        state_2 = drone_states[2]
        
        # 修复了使用 .copy() 防止多线程报错
        info_1 = (
            f"\n\n[MY STATUS] Drone 2 is at relative pos (X:{state_2['x']:.2f}, Y:{state_2['y']:.2f}, Z:{state_2['z']:.2f}, Yaw:{state_2['yaw']:.1f}°)."
            f"\n[TEAMMATE STATUS] Drone 1 is at relative pos (X:{state_1['x']:.2f}, Y:{state_1['y']:.2f}, Z:{state_1['z']:.2f}, Yaw:{state_1['yaw']:.1f}°). Landmarks found: {list(state_1['landmarks'].copy())}."
            f"\n[TEAMMATE INTENT]: {state_1['intent']}"
        )

        # 5步动态唤醒机制 (Drone 2)
        if step_drone2 % 5 == 0:
            info_1 += f"\n\n[SYSTEM REMINDER]: You have searched for {step_drone2} steps. If you still have not found the target '{current_instruction}', you might be stuck in a blind spot. Please immediately execute a large-angle rotation (e.g., rotate_cw(90) or rotate_ccw(120)) to explore entirely new areas!"

        
        drone2_messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{last_image_str_drone2}"}},
                {"type": "text", "text": current_instruction + info_1}
            ]
        })

        output_text_drone2, drone2_messages = ask(drone2_messages)
        print('DRONE 2 Assistant:', output_text_drone2)

        append_client_log(2, step_drone2, output_text_drone2)
        step_drone2 += 1
        update_virtual_state(2, output_text_drone2)
        send_command(drone_2, output_text_drone2)
        
        for text in ['break', 'drone.land()']:
            if text in output_text_drone2:
                is_drone2_stop = True
                
        drone2_messages = delete_image(drone2_messages)
        
        if is_first_message:
            sleep(1)
            is_first_message = False


# ====== 重构：完全依赖物理坐标的解析与映射 ======
def handle_client_telemetry(drone_id, conn):
    buffer = ""
    while True:
        try:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                break
            buffer += data
            
            # 处理粘包问题，按行解析
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                
                if line.startswith("TELE:"):
                    state_str = line.replace("TELE:", "")
                    
                    state_dict = {}
                    for item in state_str.split(';'):
                        if ':' in item:
                            k, v = item.split(':', 1)
                            try:
                                state_dict[k.strip()] = float(v.strip())
                            except ValueError:
                                pass 
                    
                    # 接收 Client 传来的 px, py, tof, yaw
                    if all(k in state_dict for k in ('px', 'py', 'tof', 'yaw')):
                        mvo_px = state_dict['px']
                        mvo_py = state_dict['py']
                        tof = state_dict['tof'] 
                        imu_yaw = state_dict['yaw']
                        
                        state = drone_states[drone_id]
                        
                        # 记录飞机第一次连上时的误差偏移量 (Offset)
                        if 'mvo_offset_x' not in state:
                            state['mvo_offset_x'] = mvo_px
                            state['mvo_offset_y'] = mvo_py
                            state['init_yaw'] = state['yaw'] - imu_yaw 
                        
                        # 计算相对于起飞点的纯位移 (Delta)
                        # 在 Tello 的光流坐标系中：通常 px 对应前后，py 对应左右
                        # 在你的全局坐标中，如果想让左右等同于 X，前后等同于 Y，可以直接对调映射
                        delta_x = mvo_py - state['mvo_offset_y'] 
                        delta_y = mvo_px - state['mvo_offset_x'] 
                        
                        # 加上全局设定的基准点 (base_x, base_y)，得到最终的全局坐标
                        state['x'] = state['base_x'] + delta_x
                        state['y'] = state['base_y'] + delta_y
                        state['z'] = tof / 100.0
                        state['yaw'] = (state['init_yaw'] + imu_yaw) % 360
                        
        except Exception as e:
            print(f"Telemetry lost for Drone {drone_id}: {e}")
            break
# ==========================================================

def accept_connections(server_sock):
    global drone_1, drone_2, clients
    global is_drone1_stop, is_drone2_stop, drone1_messages, drone2_messages, current_instruction
    while True:
        try:
            conn, addr = server_sock.accept()
            print(f"Client connected from {addr}")
            
            if addr[0] == "10.113.163.114":
                drone_1 = TelloPyServer(f"{addr[0]}:{addr[1]}", 'orin_1', conn, 5001, SERVER_IP)
                print("Server drone for orin_1 created.")

                # --- 启动 1 号机的物理数据接收线程 ---
                threading.Thread(target=handle_client_telemetry, args=(1, conn), daemon=True).start()
                # ----------------------------------------

                # 热插拔：连上时如果已经有全局任务，立即加入
                if current_instruction is not None:
                    drone1_messages = get_message_template()
                    is_drone1_stop = False
                    
            elif addr[0] == "10.113.163.121":
                drone_2 = TelloPyServer(f"{addr[0]}:{addr[1]}", 'orin_2', conn, 5002, SERVER_IP)
                print("Server drone for orin_2 created.")

                # --- 启动 2 号机的物理数据接收线程 ---
                threading.Thread(target=handle_client_telemetry, args=(2, conn), daemon=True).start()
                # ----------------------------------------

                # 热插拔：连上时如果已经有全局任务，立即加入
                if current_instruction is not None:
                    drone2_messages = get_message_template()
                    is_drone2_stop = False
                    
            clients.append((conn, addr))
        except Exception as e:
            print(f"Accept connections error: {e}")
            break

def main():
    global drone_1, drone_2
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((SERVER_IP, SERVER_PORT))
    server_sock.listen(2)

    print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
    print("Waiting for drones to connect... (Gradio UI is starting)")

    # 1. 开启后台网络接收线程 (允许热插拔)
    accept_thread = Thread(target=accept_connections, args=(server_sock,), daemon=True)
    accept_thread.start()

    # 2. 启动画面监控线程
    video_thread = Thread(target=video, daemon=True)
    video_thread.start()

    # 3. 启动双机独立决策
    worker1 = Thread(target=drone1_worker_loop, daemon=True)
    worker2 = Thread(target=drone2_worker_loop, daemon=True)
    worker1.start()
    worker2.start()

    with gr.Blocks() as ui:
        gr.Markdown("# Drone Command Center")

        with gr.Row():
            instruction_input = gr.Textbox(label="Instruction", placeholder="Enter command...")

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Row():
            drone1_out = gr.Textbox(
                label="Drone 1 Messages",
                value=lambda: get_drone_messages(1),
                every=1,
                lines=10
            )
            drone2_out = gr.Textbox(
                label="Drone 2 Messages",
                value=lambda: get_drone_messages(2),
                every=1,
                lines=10
            )

        submit_btn.click(
            fn=process_instruction,
            inputs=instruction_input,
            outputs=None
        )

    ui.launch()

    drones = [drone_1, drone_2]

    for drone in drones:
        if drone and drone.socket:
            drone.socket.close()

    server_sock.close()



if __name__ == "__main__":
    main()