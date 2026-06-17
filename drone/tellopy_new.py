from .basic import BasicDrone
from tellopy import Tello
from time import sleep
import math

import cv2
import socket
import threading
import numpy as np

from PIL import Image
import io

VIDEO_LIVE_PROT = 5000

from aiortc.rtp import RtpPacket
from aiortc.rtcrtpparameters import RTCRtpCodecParameters

class TelloPy(BasicDrone):
    # global server_video_port
    def __init__(self, **kwargs):
        self.drone = Tello()

        self.rtp_seq = 0
        self.rtp_timestamp = 0
        self.rtp_ssrc = 12345

        self.video_loopback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # global server_video_port, server_ip
        self.server_video_port = kwargs.get('server_video_port', None)
        self.server_ip = kwargs.get('server_ip', None)
        self.drone.subscribe(
            self.drone.EVENT_FLIGHT_DATA,
            self._move_handler)

        # video part
        self.drone.start_video()
        self.drone.subscribe(
            self.drone.EVENT_VIDEO_FRAME,
            self._video_handler
        )
        self.print_flag = True
        self.current_image = None
        self.video_stop = False

        self.time_sleep = 5
        print(f"Sleep for {self.time_sleep} seconds to initialize video stream")
        sleep(self.time_sleep)

        self.telemetry_str = "" # <--- 新增：用于临时存放解析好的飞行数据





        self.video_thread = threading.Thread(
            None, self._video_thread, daemon=True
        )
        self.video_thread.start()


    def connect(self) -> bool:
        self.drone.connect()
        self.drone.wait_for_connection(60.0)
        return True

    def takeoff(self) -> bool:
        self.drone.takeoff()
        return True

    def land(self) -> bool:
        self.drone.land()
        return True

    def move_forward(self, val, time=1) -> bool:
        self.drone.forward(val)
        sleep(time)
        self.drone.forward(0)
        return True

    def move_down(self, val, time=1):
        self.drone.down(val)
        sleep(time)
        self.drone.down(0)
        return True

    def move_left(self, val, time=1):
        self.drone.left(val)
        sleep(time)
        self.drone.left(0)
        return True

    def move_right(self, val, time=1):
        self.drone.right(val)
        sleep(time)
        self.drone.right(0)
        return True

    def move_up(self, val, time=1):
        self.drone.up(val)
        sleep(time)
        self.drone.up(0)
        return True

    def move_backward(self, val, time=1):
        self.drone.backward(val)
        sleep(time)
        self.drone.backward(0)
        return True

    def rotate_cw(self, val, time=1):
        self.drone.clockwise(val)
        sleep(time)
        self.drone.clockwise(0)
        return True

    def rotate_ccw(self, val, time=1):
        self.drone.counter_clockwise(val)
        sleep(time)
        self.drone.counter_clockwise(0)
        return True

    def start_video(self):
        self.drone.start_video()
        return True

    def quit(self):
        self.drone.quit()
        self.video_stop = True
        self.video_thread.join()

        return True

    def set_zero(self):
        self.drone.clockwise(0)

    def _video_thread(self):
        try:
            cap = cv2.VideoCapture(
                f"udp://@127.0.0.1:5000"
            )
            if not cap.isOpened():
                cap.open()
            while not self.video_stop:
                # print('attempting to read video frame...')
                res, self.current_image = cap.read()
        except Exception as e:
            print(e)
        finally:
            cap.release()
            print("Video Stream stopped.")

    # def _move_handler(self, event, sender, data, **args):
    #     drone = sender
    #     # print(type(event), type(sender), type(data))
    #     if event is drone.EVENT_FLIGHT_DATA:
    #         pass

    def _move_handler(self, event, sender, data, **args):
        drone = sender
        if event is drone.EVENT_FLIGHT_DATA:
            try:
                import math # 引入数学库计算四元数
                
                # 1. 提取高度
                tof_raw = getattr(data, 'height', 15)  
                tof = tof_raw * 10  
                
                # 2. 提取底层 MVO 绝对坐标
                mvo = drone.log_data.mvo
                px = getattr(mvo, 'pos_x', 0.0) 
                py = getattr(mvo, 'pos_y', 0.0) 
                
                # 3. 提取底层 IMU 四元数并计算绝对 Yaw
                imu = drone.log_data.imu
                q0 = getattr(imu, 'q0', 1.0) # 对应 w
                q1 = getattr(imu, 'q1', 0.0) # 对应 x
                q2 = getattr(imu, 'q2', 0.0) # 对应 y
                q3 = getattr(imu, 'q3', 0.0) # 对应 z
                
                # 四元数转欧拉角 (Z轴偏航角) 公式
                yaw_rad = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
                yaw_deg = math.degrees(yaw_rad)
                
                # 发送全套物理状态给 Server
                self.telemetry_str = f"px:{px:.3f};py:{py:.3f};tof:{tof};yaw:{yaw_deg:.2f}"
            except Exception as e:
                self.telemetry_str = f"error:{e}"


    # video part

    def _video_handler(self, event, sender, data):
        if self.print_flag:
            print("Video Frame")
            print(self.server_ip, self.server_video_port)
            self.print_flag = False
        # print(data)
        # print(type(event), type(sender), type(data))
        self.video_loopback.sendto(data, ('127.0.0.1', 5000))
        if self.server_ip is not None and self.server_video_port is not None:
            # print("send video to server")
            # packet = RtpPacket(
            #     payload_type=96,  # dynamic for video
            #     sequence_number=self.rtp_seq,
            #     timestamp=self.rtp_timestamp,
            #     ssrc=self.rtp_ssrc,
            #     payload=data
            # )
            print(self.server_ip, self.server_video_port)
            self.video_loopback.sendto(data, (self.server_ip, self.server_video_port))
            # self.video_loopback.sendto(packet.serialize(), (self.server_ip, self.server_video_port))

            # self.rtp_seq += 1
            # self.rtp_timestamp += 3000  # assuming 30fps, adjust as necessary
