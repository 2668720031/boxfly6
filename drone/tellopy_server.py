from .basic import BasicDrone
from tellopy import Tello
from time import sleep

import cv2
import socket
import threading
import numpy as np

from PIL import Image
import io


VIDEO_LIVE_PROT=5000

class TelloPyServer(BasicDrone):
    def __init__(self, client_ip:str, client_id:str, bind_socket:socket.socket, video_listen_port, server_ip:str):
        # self.drone=Tello()
        # self.drone.subscribe(
        #     self.drone.EVENT_FLIGHT_DATA,
        #     _move_handler)
        #
        # # video part
        # self.drone.start_video()
        # self.drone.subscribe(
        #     self.drone.EVENT_VIDEO_FRAME,
        #     _video_handler
        # )
        self.client_ip=client_ip
        self.client_id=client_id
        self.server_ip=server_ip
        self.current_image=None
        self.video_stop=False
        self.socket=bind_socket
        self.video_listen_port = video_listen_port


        self.video_thread=threading.Thread(
            None, self._video_thread, daemon=True
        )
        self.video_thread.start()
        

        
    def connect(self) -> bool:
        pass
    
    def takeoff(self) -> bool:
        pass
    
    def land(self) -> bool:
        pass
    
    def move_forward(self, val, time=1) -> bool:
        pass
    
    def move_down(self, val, time=1):
        pass
    
    def move_left(self, val, time=1):
        pass
    
    def move_right(self, val, time=1):
        pass

    def move_up(self, val, time=1):
        pass

    def move_backward(self, val, time=1):
        pass
    
    def rotate_cw(self, val, time=1):
        pass
    
    def rotate_ccw(self, val, time=1):
        pass
    
    def start_video(self):
        pass

    def quit(self):
        pass
    
    def set_zero(self):
        pass
        
    def _video_thread(self):
        time_sleep = 7
        print(f"Sleep for {time_sleep}s for video stream preparation")
        sleep(time_sleep)
        cap = cv2.VideoCapture(
            f"udp://@{self.server_ip}:{self.video_listen_port}"
        )
        while True:
            try:
                if not cap.isOpened():
                    # cap.open(f"udp://@{self.server_ip}:{self.video_listen_port}")
                    cap.open(f"rtp://@{self.server_ip}:{self.video_listen_port}")
                while not self.video_stop:
                    res, self.current_image=cap.read()
            except Exception as e:
                print(e)
            finally:
                cap.release()

                print("Video Stream stopped.")
                break
         
# def _move_handler(event, sender, data, **args):
#     drone = sender
#     if event is drone.EVENT_FLIGHT_DATA:
#         pass

# video part
# video_loopback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# def _video_handler(event, sender, data):
#     # print("Video Frame")
#     video_loopback.sendto(data,("0.0.0.0",5000))
#     # video_loopback.sendto(data,('10.113.163.114', 5000))