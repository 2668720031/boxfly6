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

class TelloPy(BasicDrone):
    def __init__(self):
        self.drone=Tello()
        self.drone.subscribe(
            self.drone.EVENT_FLIGHT_DATA, 
            _move_handler)

        # video part
        self.drone.start_video()
        self.drone.subscribe(
            self.drone.EVENT_VIDEO_FRAME, 
            _video_handler
        )
        
        self.current_image=None
        self.video_stop=False
        
        self.video_thread=threading.Thread(
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
        self.video_stop=True
        self.video_thread.join()
        
        return True
    
    def set_zero(self):
        self.drone.clockwise(0)
        
    def _video_thread(self):
        try:
            cap= cv2.VideoCapture(
                f"udp://@127.0.0.1:5000"
            )
            if not cap.isOpened():
                cap.open()
            while not self.video_stop:
                # print('attempting to read video frame...')
                res, self.current_image=cap.read()
        except Exception as e:
            print(e)
        finally:
            cap.release()
            print("Video Stream stopped.")
         
def _move_handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        pass

# video part
video_loopback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def _video_handler(event, sender, data):
    # print("Video Frame")
    # print(data)
    # print(type(event), type(sender), type(data))
    video_loopback.sendto(data,('127.0.0.1',5000))