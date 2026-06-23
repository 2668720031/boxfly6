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