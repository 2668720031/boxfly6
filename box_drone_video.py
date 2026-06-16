from drone import TelloPy
import pygame
import cv2
import numpy as np
from threading import Thread

drone = TelloPy()

drone.connect()

def video():
    pygame.init()
    pygameWindow = pygame.display.set_mode((980,720))
    pygame.display.set_caption("Drone Camera Feed")
    clock = pygame.time.Clock()

    try:
        while True:
            try:
                frame = cv2.cvtColor(drone.current_image, cv2.COLOR_BGR2RGB)
                frame = np.rot90(frame)
                frame = np.flipud(frame)
                frame = pygame.surfarray.make_surface(frame)
                pygameWindow.fill((0,0,0))
                pygameWindow.blit(frame,(0,0))
            except:
                pygameWindow.fill((0,0,255))
            pygame.display.update()
            clock.tick(30)
    finally:
        pygame.quit()
        drone.land()
        drone.quit()

video_thread=Thread(target=video,daemon=True)
video_thread.start()