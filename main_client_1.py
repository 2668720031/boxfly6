import traceback

import socket
import threading
# import pygame
import cv2
import numpy as np
# import gradio as gr

import io
import base64

from time import sleep, time
from threading import Thread
from PIL import Image
from sympy import continued_fraction_reduce

# from drone import TelloPy
from drone.tellopy_new import TelloPy
from prompt import get_message_template
from utils import extract_code_blocks
from rich import print
from openai import OpenAI

import requests
import random

import os
from datetime import datetime
import torch
from collections import deque
from TS_CAN import TSCAN
import mediapipe as mp

# --- Server connection settings ---
SERVER_IP = '10.113.163.114'  # Orin AGX
SERVER_PORT = 9000
SERVER_VIDEO_PORT = 5001
DRONE_ID = "orin_1"

# Initialize the drone with server settings
drone = TelloPy(server_ip = SERVER_IP, server_video_port = SERVER_VIDEO_PORT)

# ======================== rPPG 核心算法模块 ========================
def calculate_bpm(signal_buffer, fps=30.0):
    """使用傅里叶变换 (FFT) 从心跳波形中提取 BPM"""
    if len(signal_buffer) < 150: # 需要至少收集约5秒的数据
        return -1.0
        
    signal = np.array(signal_buffer)
    # 去除直流分量 (去基线)
    signal = signal - np.mean(signal)
    
    # 傅里叶变换提取频谱
    fft_data = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1.0/fps)
    
    # 人的心率正常范围大概是 42 ~ 150 BPM，对应频率为 0.7Hz ~ 2.5Hz
    valid_idx = np.where((fft_freq >= 0.7) & (fft_freq <= 2.5))
    if len(valid_idx[0]) == 0:
        return -1.0
        
    valid_fft = np.abs(fft_data[valid_idx])
    valid_freqs = fft_freq[valid_idx]
    
    # 找到能量最大的频率峰值
    peak_idx = np.argmax(valid_fft)
    peak_freq = valid_freqs[peak_idx]
    
    # 频率(Hz) * 60 = 每分钟心跳次数(BPM)
    return peak_freq * 60.0

class DroneRPPG:
    def __init__(self, frame_length=10, image_size=36, model_path="model/UBFC-rPPG_TSCAN.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_length = frame_length
        self.image_size = image_size
        
        # 1. 实例化模型架构
        self.model = TSCAN(frame_depth=self.frame_length).to(self.device)
        
        # 2. 注入预训练权重
        self.load_pretrained_weights(model_path)
        
        self.model.eval()
        self.raw_frame_buffer = deque(maxlen=self.frame_length)
        
        # ====== MediaPipe 人脸检测器 ======
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)

        self.last_face_box = None       
        self.face_miss_count = 0        
        self.max_miss_tolerance = 15    

    def load_pretrained_weights(self, model_path):
        if not os.path.exists(model_path):
            print(f"[⚠️] 警告: 未找到权重文件 {model_path}，当前模型依然是随机初始化的。")
            return

        print(f"[*] 正在从 {model_path} 加载预训练权重...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 兼容性处理：如果包含 'state_dict' 键则提取，否则直接使用
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # 🌟 智能修复：移除 'module.' 前缀 (DataParallel 导致的)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
            
        # 加载权重
        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print("[✅] 专家级大脑激活成功！模型已获得心率识别能力。")
        except Exception as e:
            print(f"[❌] 权重加载失败: {e}")
            print("正在尝试使用 strict=False 进行部分加载...")
            self.model.load_state_dict(new_state_dict, strict=False)


    def process_frame(self, bgr_img):
        h_img, w_img = bgr_img.shape[:2]
        
        # MediaPipe 需要 RGB 格式的输入
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_img)
        
        curr_x, curr_y, curr_w, curr_h = 0, 0, 0, 0
        
        if not results.detections:
            self.face_miss_count += 1
            # 策略1：短暂丢失时，强行使用上一帧的记忆位置（抵抗断片）
            if self.last_face_box is not None and self.face_miss_count < self.max_miss_tolerance:
                curr_x, curr_y, curr_w, curr_h = self.last_face_box
            else:
                self.last_face_box = None
                self.raw_frame_buffer.clear() 
                return "NO_FACE"
        else:
            self.face_miss_count = 0
            # 提取置信度最高的人脸（通常是画面中心的主体）
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            
            # MediaPipe 返回的是比例 (0~1)，需要还原为实际像素坐标
            curr_x = int(bboxC.xmin * w_img)
            curr_y = int(bboxC.ymin * h_img)
            curr_w = int(bboxC.width * w_img)
            curr_h = int(bboxC.height * h_img)
            
        # ====== 指数移动平均 (EMA) 平滑抗抖动 ======
        # 让裁剪框像用了稳定器一样死死钉在人脸上
        if self.last_face_box is not None:
            px, py, pw, ph = self.last_face_box
            alpha = 0.3  # 信任新位置 30%，信任老位置 70%
            x = int(alpha * curr_x + (1 - alpha) * px)
            y = int(alpha * curr_y + (1 - alpha) * py)
            w = int(alpha * curr_w + (1 - alpha) * pw)
            h = int(alpha * curr_h + (1 - alpha) * ph)
        else:
            x, y, w, h = curr_x, curr_y, curr_w, curr_h
            
        self.last_face_box = (x, y, w, h)
        
        # 边界安全检查，防止框跑到画面外报错
        x = max(0, x)
        y = max(0, y)
        w = min(w_img - x, w)
        h = min(h_img - y, h)

        # 裁剪并预处理
        face_roi = bgr_img[y:y+h, x:x+w]
        if face_roi.size == 0 or w < 10 or h < 10: 
            return "NO_FACE"
            
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (self.image_size, self.image_size))
        face_norm = face_resized.astype(np.float32) / 255.0 
        
        self.raw_frame_buffer.append(face_norm)
        
        if len(self.raw_frame_buffer) == self.frame_length:
            return self.infer()
        return "BUFFERING"

    def infer(self):
        # 推理逻辑完全不变
        raw_frames = np.array(self.raw_frame_buffer) 
        diff_frames = np.zeros_like(raw_frames)
        diff_frames[1:] = raw_frames[1:] - raw_frames[:-1] 
        
        concat_frames = np.concatenate((diff_frames, raw_frames), axis=-1)
        concat_frames = np.transpose(concat_frames, (0, 3, 1, 2))
        
        input_tensor = torch.tensor(concat_frames, dtype=torch.float32).contiguous().to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor) 
            
        return output.cpu().numpy()

def video():
    print(f"Video ({DRONE_ID})")
    # It's good practice to create the window once
    cv2.namedWindow("Drone Camera", cv2.WINDOW_NORMAL)

    try:
        while True:
            # 1. Check for 'q' at the very start of every loop iteration
            # This ensures the OS gets a heartbeat even if the image is None
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            current_bgr_img = drone.current_image

            if current_bgr_img is None:
                # If no image, we just loop back and waitKey(1) again
                continue

            cv2.imshow("Drone Camera", current_bgr_img)

    except Exception as e:
        print(f"Video Error: {e}")

    finally:
        cv2.destroyAllWindows()
        # Be careful: calling drone.land() here will trigger every time
        # the video window closes or crashes!
        drone.land()
        drone.quit()

def perform_instruction(output_text: str):
    if not output_text:
        return
    print(f"Performing instruction: {output_text}")

    if 'break' in output_text:
        return
    res = extract_code_blocks(output_text)

    # drone.set_zero()
    if len(res) > 0:
        print(res)
        for i in range(len(res)):
            # drone.set_zero()
            if "land" in res[i]['code']:
                print("Land")
                drone.land()
                break
            exec(res[0]['code'])
            # sleep(0.5)

    if 'drone.land' in output_text:
        return

# ====== 直接提取底层物理变量回传 ======
def telemetry_reporter(sock, drone_obj):
    """
    直接从 drone_obj 读取实时更新的物理属性，组装后发给 Server。
    """
    while True:
        try:
            # 组装格式： px:值; py:值; tof:值; yaw:值\n
            telemetry_str = f"px:{drone_obj.mvo_px:.3f};py:{drone_obj.mvo_py:.3f};tof:{drone_obj.tof};yaw:{drone_obj.imu_yaw:.2f}"
            sock.sendall(f"TELE:{telemetry_str}\n".encode('utf-8'))
        except socket.error as e:
            print(f"Telemetry socket connection lost: {e}")
            break # Socket 断开则退出线程
        except Exception as e:
            pass # 忽略其他解析错误，保证线程不死
            
        sleep(0.05) # 20Hz 更新频率
# ===============================================

# ====== 完美日志版 rPPG 线程 ======
def rppg_worker(sock, drone_obj):
    print(f"\n[*] =======================================")
    print(f"[*] rPPG Worker 启动，进入全透视 Debug 模式")
    print(f"[*] =======================================\n")
    
    rppg = DroneRPPG(frame_length=10, image_size=36)
    pulse_buffer = deque(maxlen=150) 
    last_calc_time = time()
    
    # 日志初始化逻辑
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, 'rPPG1.txt')
    
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"=========================================\n")
        f.write(f"🚀 新的 rPPG 测试会话开始: {start_time_str}\n")
        f.write(f"=========================================\n")

    # 使用列表包装状态，完美绕过 Python 作用域导致的报错
    last_logged_state = [""]

    def write_log(msg, force=False):
        state_key = msg.split(':')[0] if ':' in msg else msg
        if force or state_key != last_logged_state[0]:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            last_logged_state[0] = state_key

    frame_counter = 0 
    
    while True:
        try:
            if drone_obj is None or drone_obj.current_image is None:
                sleep(0.1)
                continue
                
            frame = drone_obj.current_image.copy()
            out = rppg.process_frame(frame)
            frame_counter += 1
            
            # 🌟 核心修复：增加 isinstance 类型判断，防止 Numpy 数组与字符串比较引发歧义报错
            if isinstance(out, str):
                if out == "NO_FACE":
                    debug_msg = "[Debug] 未检测到人脸，请正对摄像头、调整光线或距离..."
                    print(f"[帧 {frame_counter}] ❌ {debug_msg}")
                    write_log(debug_msg)
                    
                elif out == "BUFFERING":
                    debug_msg = f"[Debug] 捕捉到人脸！TS-CAN 正在预热: {len(rppg.raw_frame_buffer)}/10"
                    print(f"[帧 {frame_counter}] ⏳ {debug_msg}")
                    if len(rppg.raw_frame_buffer) in [1, 5, 9]:
                        write_log(debug_msg)
            
            # 如果 out 不是字符串且不为 None，说明它是一个包含波形数据的 Numpy 数组
            elif out is not None:
                val = float(np.mean(out[-1])) 
                pulse_buffer.append(val)
                
                print(f"[帧 {frame_counter}] 📈 正在收集脉搏波... {len(pulse_buffer)}/150 帧")
                
                if len(pulse_buffer) % 30 == 0:
                    write_log(f"[Debug] 正在持续收集面部脉搏波... 进度 {len(pulse_buffer)}/150 帧")
                
                if len(pulse_buffer) == 150 and (time() - last_calc_time) > 1.5:
                    bpm = calculate_bpm(pulse_buffer, fps=30.0)
                    
                    if bpm > 0:
                        log_msg = f"🎯 视觉捕捉心率成功: {bpm:.1f} BPM"
                        print(f"\n🚀🚀🚀 {log_msg} 🚀🚀🚀\n")
                        write_log(log_msg, force=True) 
                        sock.sendall(f"TELE:hr:{bpm:.1f}\n".encode('utf-8'))
                    else:
                        fail_msg = f"[Debug] 凑齐了150帧，但信号太弱无法计算BPM"
                        print(fail_msg)
                        write_log(fail_msg, force=True)
                        
                    last_calc_time = time()
            
            sleep(0.033)
            
        except Exception as e:
            # 强化报错打印：将错误直接写入日志，让你随时能查阅
            err_msg = f"🚨 [Fatal Error] rPPG 线程崩溃: {e}"
            print(f"\n{err_msg}\n")
            write_log(err_msg, force=True)
            sleep(1)

def main():
    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))
    print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}")
    # sock.sendall(DRONE_ID.encode())

    # Start thread to listen for server commands
    # threading.Thread(target=listen_server, args=(sock,), daemon=True).start()

    # Start video and drone as before
    video_thread = Thread(target=video, daemon=True)
    video_thread.start()
    drone.connect()
    drone.takeoff()

    # ====== 启动遥测数据回传线程，正确传入 drone 对象 ======
    threading.Thread(target=telemetry_reporter, args=(sock, drone), daemon=True).start()
    # =========================================

    # ====== 启动 rPPG 视觉感知心率线程 ======
    threading.Thread(target=rppg_worker, args=(sock, drone), daemon=True).start()
    # =========================================

    # Keep the client running
    try:
        while True:
            try:
                data = sock.recv(10240).decode()
                # if not data:
                #     break
                if not data:
                    continue

                if "shutdown" in data:
                    print("shutting down")
                    break

                print(f"Received data from server: {data}")
                perform_instruction(data)
                # sock.sendall("finish".encode())
            except Exception as e:
                print(f"Connection error: {e}")
                break

    except Exception as e:
        print(e)
    finally:
        drone.land()
        drone.quit()
        sock.close()


if __name__ == "__main__":
    main()