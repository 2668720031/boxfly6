import pygame
import time

### vvv IMPORTANT BIT vvv ###
# Minimum you need to get video from tello using tellopy

import socket
import threading
import numpy as np
import cv2
import tellopy

vidOut = None  # Frame that can be used externally

stop_cam = False  # Stop flag
cam_error = None  # Error message can view or raise()
loopback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# loopback.bind(('127.0.0.123',1234))

def cam():  # RUN THE WHILE LOOP AS FAST AS POSSIBLE!
    ##           VIDEO DISTORTION OTHERWISE! Use vidOut buffer :)
    try:
        global vidOut
        global stop_cam
        global cam_error
        cap = cv2.VideoCapture("udp://@127.0.0.1:5000")  # Random address
        if not cap.isOpened:
            cap.open()
        while not stop_cam:
            res, vidOut = cap.read()
    except Exception as e:
        cam_error = e
    finally:
        cap.release()
        print("Video Stream stopped.")


camt = threading.Thread(None, cam)  # Start thread
camt.start()


def videoFrameHandler(event, sender, data):  # Video Frame loopback
    loopback.sendto(data, ('127.0.0.1', 5000))  # Random address


drone = tellopy.Tello()
drone.connect()
drone.start_video()
drone.subscribe(drone.EVENT_VIDEO_FRAME, videoFrameHandler)

### ^^^ IMPORTANT BIT ^^^ ###

### vvv Insignificant test code vvv ###
"""
    Simple stuff. For testing. Add your control code here.
    Right now, only these buttons work:
        SPACE  - Quit
        RETURN - Take JPEG and put in the same folder
                 as this file is in (probably)
        z      - Toggle 4:3 960x720 or 16:9 1280x720

    You can also display video to pygame by doing:
        frame = cv2.cvtColor(vidOut, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frame = pygame.surfarray.make_surface(frame)
        pygameWindow.blit(frame,(0,0))
    Just make sure the pygame window fits the frame
    or rezise to fit.
"""


def flightDataHandler(event, sender, data):
    # print(data)
    pass


def handleFileReceived(event, sender, data):
    global date_fmt
    # Create a file in the same folder as program (hopefully)
    path = 'tello' + str(int(time.time())) + '.jpeg'
    with open(path, 'wb') as fd:
        fd.write(data)


drone.subscribe(drone.EVENT_FILE_RECEIVED, handleFileReceived)
drone.subscribe(drone.EVENT_FLIGHT_DATA, flightDataHandler)

pygame.init()
pygameWindow = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("YAY!")
clock = pygame.time.Clock()

try:
    done = False
    while not done:
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    done = True
                if e.key == pygame.K_RETURN:
                    drone.take_picture()
                if e.key == pygame.K_z:
                    drone.set_video_mode(not drone.zoom)

        try:
            frame = cv2.cvtColor(vidOut, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            pygameWindow.fill((0, 0, 0))
            pygameWindow.blit(frame, (0, 0))
        except:
            pygameWindow.fill((0, 0, 255))

        pygame.display.update()
        clock.tick(30)
finally:
    stop_cam = True
    camt.join()
    drone.quit()
    pygame.quit()
    cv2.destroyAllWindows()
    time.sleep(3)
    print("END")