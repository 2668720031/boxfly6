from drone import TelloPy
import time

drone = TelloPy()

drone.takeoff()

input()

drone.move_left(30)

input()

drone.move_right(30)

input()

drone.move_up(30)

input()

drone.move_down(30)

input()

