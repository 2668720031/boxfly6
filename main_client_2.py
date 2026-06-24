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
SERVER_VIDEO_PORT = 5002
DRONE_ID = "orin_2"

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
    def __init__(self, frame_length=10, image_size=36):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_length = frame_length
        self.image_size = image_size
        self.model = TSCAN(frame_depth=self.frame_length).to(self.device)
        self.model.eval()
        self.raw_frame_buffer = deque(maxlen=self.frame_length)
        
        # ====== MediaPipe 人脸检测器 ======
        self.mp_face_detection = mp.solutions.face_detection
        # model_selection=0 适合 2 米以内的近距离人脸（无人机悬停距离）
        # min_detection_confidence=0.5 稍微放宽置信度，增加抗恶劣光线能力
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)

        # 追踪与平滑状态变量
        self.last_face_box = None       
        self.face_miss_count = 0        
        self.max_miss_tolerance = 15    # 容忍最多连续丢失 15 帧 (约0.5秒)

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


    # pass

# ====== 重写：直接提取底层物理变量回传 ======
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
    print(f"[*] rPPG Worker 启动，等待获取无人机画面...")
    rppg = DroneRPPG(frame_length=10, image_size=36)
    
    pulse_buffer = deque(maxlen=150) 
    last_calc_time = time()
    
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, 'rPPG1.txt')
    
    # 🌟 核心修改 1：最开头打开文件用 'w' (Write)，每次重新运行脚本都会清空旧文件！
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"=========================================\n")
        f.write(f"🚀 新的 rPPG 测试会话开始: {start_time_str}\n")
        f.write(f"=========================================\n")

    # 用于记录上一次写入日志的状态，防止高频刷屏
    last_logged_state = ""

    def write_log(msg, force=False):
        """智能日志写入器：运行期间使用 'a' 追加，并自动过滤重复的高频刷屏"""
        nonlocal last_logged_state
        # 提取核心状态词作为标识 (比如提取 "[Debug] 未检测到人脸")
        state_key = msg.split(':')[0] if ':' in msg else msg
        
        # 🌟 核心修改 2：只有状态发生改变，或者强制要求记录时才往 txt 里写
        if force or state_key != last_logged_state:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            last_logged_state = state_key

    while True:
        try:
            if drone_obj is None or drone_obj.current_image is None:
                sleep(0.1)
                continue
                
            frame = drone_obj.current_image.copy()
            out = rppg.process_frame(frame)
            
            if out == "NO_FACE":
                debug_msg = "[Debug] 未检测到人脸，请正对摄像头、调整光线或距离..."
                print(f"{debug_msg}          ", end='\r')
                write_log(debug_msg)  # 写入日志
                
            elif out == "BUFFERING":
                debug_msg = f"[Debug] 捕捉到人脸！TS-CAN 正在预热: {len(rppg.raw_frame_buffer)}/10"
                print(f"{debug_msg}     ", end='\r')
                # 预热阶段挑几个关键节点写入日志，证明没有卡死
                if len(rppg.raw_frame_buffer) in [1, 5, 9]:
                    write_log(debug_msg)
                    
            elif out is not None:
                val = float(np.mean(out[-1])) 
                pulse_buffer.append(val)
                
                print(f"[Debug] 正在持续收集面部脉搏波... {len(pulse_buffer)}/150 帧      ", end='\r')
                
                # 🌟 核心修改 3：在收集数据的 5 秒钟里，每收集 30 帧 (约1秒) 往日志打个卡
                if len(pulse_buffer) % 30 == 0:
                    write_log(f"[Debug] 正在持续收集面部脉搏波... 进度 {len(pulse_buffer)}/150 帧")
                
                if len(pulse_buffer) == 150 and (time() - last_calc_time) > 1.5:
                    bpm = calculate_bpm(pulse_buffer, fps=30.0)
                    
                    print("") # 换行，防止最终结果被控制台的 \r 吃掉
                    if bpm > 0:
                        log_msg = f"🎯 视觉捕捉心率成功: {bpm:.1f} BPM"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")
                        # 强制把最终成功的心率结果追加进 txt
                        write_log(log_msg, force=True) 
                        
                        sock.sendall(f"TELE:hr:{bpm:.1f}\n".encode('utf-8'))
                    else:
                        fail_msg = f"[Debug] 凑齐了150帧，但信号太弱无法计算BPM (画面过暗或人体晃动剧烈)"
                        print(fail_msg)
                        write_log(fail_msg, force=True)
                        
                    last_calc_time = time()
            
            sleep(0.033)
            
        except Exception as e:
            print(f"\r\nrPPG 线程遇到警告: {e}")
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
    
    # ====== 修复：启动遥测数据回传线程，正确传入 drone 对象 ======
    threading.Thread(target=telemetry_reporter, args=(sock, drone), daemon=True).start()
    # =========================================

    threading.Thread(target=rppg_worker, args=(sock, drone), daemon=True).start()

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