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
        
        # ===== 新增：初始化物理状态变量 =====
        self.mvo_px = 0.0
        self.mvo_py = 0.0
        self.imu_yaw = 0.0
        self.tof = 0.0

        # ===== 关键修复：订阅两个事件 =====
        self.drone.subscribe(
            self.drone.EVENT_FLIGHT_DATA,
            self._flight_data_handler)
            
        self.drone.subscribe(
            self.drone.EVENT_LOG_DATA,
            self._log_data_handler)

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

    def _flight_data_handler(self, event, sender, data, **args):
        try:
            # 基础数据里的 height 精度较低，乘以 10 转换为厘米或毫米标准
            self.tof = getattr(data, 'height', self.tof / 10.0) * 10.0
        except Exception:
            pass

    def _log_data_handler(self, event, sender, data, **args):
        try:
            # 这里的 data 是 LogData 对象，内部包含了 mvo 和 imu
            if hasattr(data, 'mvo'):
                self.mvo_px = getattr(data.mvo, 'pos_x', self.mvo_px)
                self.mvo_py = getattr(data.mvo, 'pos_y', self.mvo_py)
            
            if hasattr(data, 'imu'):
                q0 = getattr(data.imu, 'q0', 1.0) # 对应 w，默认值为 1.0 表示无旋转
                q1 = getattr(data.imu, 'q1', 0.0) # 对应 x
                q2 = getattr(data.imu, 'q2', 0.0) # 对应 y
                q3 = getattr(data.imu, 'q3', 0.0) # 对应 z
                
                # 四元数转欧拉角 (Z轴偏航角) 公式
                yaw_rad = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
                self.imu_yaw = math.degrees(yaw_rad)
        except Exception as e:
            pass

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