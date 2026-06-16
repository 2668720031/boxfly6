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

# --- Server connection settings ---
SERVER_IP = '10.113.163.114'  # Orin AGX
SERVER_PORT = 9000
SERVER_VIDEO_PORT = 5002
DRONE_ID = "orin_2"

# Initialize the drone with server settings
drone = TelloPy(server_ip = SERVER_IP, server_video_port = SERVER_VIDEO_PORT)

# # Initialize the drone with server settings
# drone = TelloPy(server_ip = SERVER_IP, server_video_port = SERVER_VIDEO_PORT)

# def video():
#     print(f"Video ({DRONE_ID})")
#     # cv2.namedWindow("Drone Camera")
#     # cv2.resizeWindow("Drone Camera", 980, 720)
#     # frame_skip = 300
#     try:
#         while True:
#             try:
#                 # if 0 < frame_skip:
#                 #     frame_skip = frame_skip - 1
#                 #     continue
#                 current_bgr_img = drone.current_image
#                 # print(current_bgr_img)
#                 if current_bgr_img is None:
#                     continue
#
#                 cv2.imshow("Drone Camera", current_bgr_img)
#
#                 if cv2.waitKey(30) & 0xFF == ord('q'):
#                     break
#
#             except Exception as e:
#                 print(e)
#
#
#     finally:
#         cv2.destroyAllWindows()
#         drone.land()
#         drone.quit()

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

# # ====== 修改后的底层遥测数据监听与回传线程 ======
# def telemetry_reporter(sock):
#     """
#     监听 Tello 默认的 8890 端口，获取光流速度和姿态，
#     并通过 TCP 实时发送给 Server。
#     加入了端口复用机制，防止与底层库冲突。
#     """
#     udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
#     # 开启端口复用（黑客技巧：允许我们和 tellopy_new 同时监听 8890 端口）
#     udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     if hasattr(socket, 'SO_REUSEPORT'):
#         try:
#             udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
#         except AttributeError:
#             pass # 某些旧版 Windows 可能不支持 REUSEPORT，忽略即可

#     try:
#         udp_sock.bind(('0.0.0.0', 8890))
#         udp_sock.settimeout(1.0)
#     except OSError as e:
#         print(f"⚠️ [警告] 无法监听 8890 端口: {e}。物理坐标更新可能会失败！")
#         return # 如果实在绑不上，优雅退出线程，不至于让客户端崩溃
    
#     while True:
#         try:
#             data, _ = udp_sock.recvfrom(1024)
#             state_str = data.decode('utf-8')
#             # state_str 的格式大概是: "pitch:0;roll:0;yaw:45;vgx:10;vgy:0;tof:120;..."
#             sock.sendall(f"TELE:{state_str}\n".encode('utf-8'))
#             sleep(0.05) # 控制一下回传频率，大概 20Hz
#         except socket.timeout:
#             continue
#         except Exception as e:
#             pass
# # ===============================================

# ====== 修改后：直接提取底层变量回传 ======
def telemetry_reporter(sock, drone_obj):
    """
    不再监听端口，直接从 drone_obj (TelloPy实例) 中读取组装好的遥测字符串
    """
    while True:
        try:
            if hasattr(drone_obj, 'telemetry_str') and drone_obj.telemetry_str:
                # 只有当非报错时，才发给 server
                if not drone_obj.telemetry_str.startswith("error"):
                    sock.sendall(f"TELE:{drone_obj.telemetry_str}\n".encode('utf-8'))
        except Exception as e:
            print(f"Telemetry socket error: {e}")
            break # Socket 断开则退出线程
        sleep(0.05) # 20Hz 更新频率
# ===============================================

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
    # ====== 新增：启动遥测数据回传线程 ======
    threading.Thread(target=telemetry_reporter, args=(sock,), daemon=True).start()
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