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

# ====== v3 MODIFIED: Initialize Log Directory and Helper Function ======
LOG_DIR = 'log'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

CLIENT1_LOG_FILE = os.path.join(LOG_DIR, 'client1.txt')
CLIENT2_LOG_FILE = os.path.join(LOG_DIR, 'client2.txt')

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
drone_states = {
    1: { # 对应 drone_1
        "x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0, # 虚拟物理坐标 (单位:米, 角度:度)
        "landmarks": set()                        # 已经搜索过的地标集合
    },
    2: { # 对应 drone_2
        "x": 1.0, "y": 0.0, "z": 1.5, "yaw": 0.0, # 假设起飞时 drone_2 在 drone_1 右侧 1.0 米处
        "landmarks": set()
    }
}

# ====== 修改点 1：定义全局变量（用于全异步控制） ======
current_instruction = None  # 存储当前网页下发的全局任务指令（如 'rack'）
is_drone1_stop = True       # 1号机的任务运行状态（初始为停止，下发任务时激活）
is_drone2_stop = True       # 2号机的任务运行状态

step_drone1 = 1
step_drone2 = 1
# =======================================================

def update_virtual_state(drone_id: int, llm_output: str):
    """
    从大模型的输出中提取地标和动作，更新全局状态 (drone_states)
    """
    state = drone_states[drone_id]
    
    # 1. 提取地标 (匹配 [Landmark]: 后面的内容，忽略大小写)
    landmark_match = re.search(r'\[Landmark\]:\s*(.+)', llm_output, re.IGNORECASE)
    if landmark_match:
        items = landmark_match.group(1).split(',')
        for item in items:
            cleaned_item = item.strip().lower()
            if cleaned_item and cleaned_item != 'none':
                state["landmarks"].add(cleaned_item)
                
    # 2. 提取动作并推算坐标 (匹配 drone.xxx(val) 格式)
    action_match = re.search(r'drone\.([a-z_]+)\((\d+)\)', llm_output)
    if not action_match:
        return
        
    action = action_match.group(1)
    try:
        val = int(action_match.group(2))
    except ValueError:
        val = 0
        
    yaw_rad = math.radians(state["yaw"])
    
    # 线性映射系数假设 (具体准确的数值还未确定)
    K_MOVE = 0.01  # 假设 1点油门力度 = 1厘米 (0.01米)
    K_ROT = 1.5    # 假设 1点油门力度 = 1.5度旋转
    K_ALT = 0.01   # 高度系数
    
    if action == "move_forward":
        state["x"] += (val * K_MOVE) * math.sin(yaw_rad)
        state["y"] += (val * K_MOVE) * math.cos(yaw_rad)
    elif action == "move_backward":
        state["x"] -= (val * K_MOVE) * math.sin(yaw_rad)
        state["y"] -= (val * K_MOVE) * math.cos(yaw_rad)
    elif action == "move_left":
        state["x"] -= (val * K_MOVE) * math.cos(yaw_rad)
        state["y"] += (val * K_MOVE) * math.sin(yaw_rad)
    elif action == "move_right":
        state["x"] += (val * K_MOVE) * math.cos(yaw_rad)
        state["y"] -= (val * K_MOVE) * math.sin(yaw_rad)
    elif action == "move_up":
        state["z"] += (val * K_ALT)
    elif action == "move_down":
        state["z"] -= (val * K_ALT)
    elif action == "rotate_cw":
        state["yaw"] = (state["yaw"] + val * K_ROT) % 360
    elif action == "rotate_ccw":
        state["yaw"] = (state["yaw"] - val * K_ROT) % 360
        
    print(f"[Drone {drone_id} State] Pos:(X:{state['x']:.2f}, Y:{state['y']:.2f}, Z:{state['z']:.2f}, Yaw:{state['yaw']:.1f}°), Landmarks: {list(state['landmarks'])}")
# ----------------------------------------


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
        with open(CLIENT1_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"========== NEW INSTRUCTION: {instruction} ==========\n\n")
        with open(CLIENT2_LOG_FILE, 'a', encoding='utf-8') as f:
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


# def send_command(drone: TelloPyServer, output_text: str):

#     # instructions = process_instruction()
#     # clients[drone_id].sendall(output_text.encode())
#     drone.socket.sendall(output_text.encode())

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


# def video():
#     global drone_1, drone_2, first_frame_flag
#     print("Video thread started.")

#     # Create the window once outside the loop
#     cv2.namedWindow("Merged Drone Cameras", cv2.WINDOW_NORMAL)

#     try:
#         while True:
#             # 1. ALWAYS run waitKey to keep the OS happy and process window events
#             # This is what prevents the "Not Responding" popup.
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             if drone_1 is None or drone_2 is None:
#                 sleep(1)
#                 continue

#             try:
#                 img_1 = drone_1.current_image
#                 img_2 = drone_2.current_image

#                 if img_1 is None or img_2 is None:
#                     continue

#                 img_merged = np.hstack((img_1, img_2))
#                 img_resize_merged = cv2.resize(img_merged, (0, 0), fx=0.8, fy=0.8)
#                 cv2.imshow("Merged Drone Cameras", img_resize_merged)

#             except Exception as e:
#                 print(f"Frame processing error: {e}")
#                 # We don't 'break' here, we just wait for the next frame

#     except Exception as e:
#         print(f"Thread error: {e}")
#     finally:
#         cv2.destroyAllWindows()
#         print("Video thread closed.")

# ====== 修改为支持热插拔与单机显示的动态视频流 ======
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
# =======================================================


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
        
        info_2 = (
            f"\n\n[MY STATUS] Drone 1 is at relative pos (X:{state_1['x']:.2f}, Y:{state_1['y']:.2f}, Z:{state_1['z']:.2f}, Yaw:{state_1['yaw']:.1f}°)."
            f"\n[TEAMMATE STATUS] Drone 2 is at relative pos (X:{state_2['x']:.2f}, Y:{state_2['y']:.2f}, Z:{state_2['z']:.2f}, Yaw:{state_2['yaw']:.1f}°). Landmarks found: {list(state_2['landmarks'])}."
        )
        
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
        
        info_1 = (
            f"\n\n[MY STATUS] Drone 2 is at relative pos (X:{state_2['x']:.2f}, Y:{state_2['y']:.2f}, Z:{state_2['z']:.2f}, Yaw:{state_2['yaw']:.1f}°)."
            f"\n[TEAMMATE STATUS] Drone 1 is at relative pos (X:{state_1['x']:.2f}, Y:{state_1['y']:.2f}, Z:{state_1['z']:.2f}, Yaw:{state_1['yaw']:.1f}°). Landmarks found: {list(state_1['landmarks'])}."
        )
        
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


# def main():
#     global drone_1, drone_2
#     server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server_sock.bind((SERVER_IP, SERVER_PORT))
#     server_sock.listen(2)
#     print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
#     print(f"Waiting for drones to connect")

#     # threading.Thread(target=accept_clients, args=(server_sock,), daemon=True).start()
#     # send_command()
#     while len(clients) < 2:
#         # server_sock.settimeout(20)
#         conn, addr = server_sock.accept()
#         print(f"Client connected from {addr}")
#         if addr[0] == "10.113.163.114":
#             drone_1 = TelloPyServer(f"{addr[0]}:{addr[1]}", 'orin_1', conn, 5001, SERVER_IP)
#             print("Server drone for orin_1 created.")

#         # todo: orin_2 ip
#         elif addr[0] == "10.113.163.121":
#             drone_2 = TelloPyServer(f"{addr[0]}:{addr[1]}", 'orin_2', conn, 5002, SERVER_IP)
#             print("Server drone for orin_2 created.")
#         clients.append((conn, addr))


#     print("All drones connected, Starting video ")
#     video_thread = Thread(target=video, daemon=True)
#     video_thread.start()
#     # UI launcher on server
#     # with gr.Blocks() as ui:
#     #     gr.Interface(
#     #         process_instruction,
#     #         'textbox',
#     #         None
#     #     )
#     #     gr.Textbox(
#     #         label="Assistant Message List",
#     #         value=get_last_assistant_message,
#     #         every=1,
#     #     )
#     with gr.Blocks() as ui:
#         gr.Markdown("# Drone Command Center")

#         with gr.Row():
#             # Input area
#             instruction_input = gr.Textbox(label="Instruction", placeholder="Enter command...")

#         with gr.Row():
#             submit_btn = gr.Button("Submit", variant="primary")
#             clear_btn = gr.Button("Clear")

#         with gr.Row():
#             # Two output textboxes side-by-side
#             drone1_out = gr.Textbox(
#                 label="Drone 1 Messages",
#                 value=lambda: get_drone_messages(1),
#                 every=1,
#                 lines=10
#             )
#             drone2_out = gr.Textbox(
#                 label="Drone 2 Messages",
#                 value=lambda: get_drone_messages(2),
#                 every=1,
#                 lines=10
#             )

#         # Link the submit button to your processing function
#         # Note: process_instruction must exist in your scope
#         submit_btn.click(
#             fn=process_instruction,
#             inputs=instruction_input,
#             outputs=None
#         )

#     ui.launch()

#     drones = [drone_1, drone_2]

#     # Exit Logic
#     for drone in drones:
#         # drone.socket.sendall("shutdown".encode())
#         drone.socket.close()

#         # del drone

#     server_sock.close()

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
                # 热插拔：连上时如果已经有全局任务，立即加入
                if current_instruction is not None:
                    drone1_messages = get_message_template()
                    is_drone1_stop = False
                    
            elif addr[0] == "10.113.163.121":
                drone_2 = TelloPyServer(f"{addr[0]}:{addr[1]}", 'orin_2', conn, 5002, SERVER_IP)
                print("Server drone for orin_2 created.")
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