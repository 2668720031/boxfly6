from .basic import BasicDrone
from utils import print

class VirtualDrone(BasicDrone):
    def __init__(self):
        pass
    
    def connect(self) -> bool:
        print('Successfully connected to virtual drone')
        return True
    
    def takeoff(self) -> bool:
        print('Taking off virtual drone')
        return True
    
    def land(self) -> bool:
        print('Landing virtual drone')
        return True
    
    def start_stream(self) -> bool:
        pass
    
    def stop_stream(self) -> bool:
        pass
    
    def get_frame(self):
        pass
    
    def move_forward(self, distance) -> bool:
        print(f'Moving forward {distance} cm')
        return True
    
    def move_backward(self, distance) -> bool:
        print(f'Moving backward {distance} cm')
        return True
    
    def move_left(self, distance) -> bool:
        print(f'Moving left {distance} cm')
        return True
    
    def move_right(self, distance) -> bool:
        print(f'Moving right {distance} cm')
        return True
    
    def move_up(self, distance) -> bool:
        print(f'Moving up {distance} cm')
        return True
    
    def move_down(self, distance) -> bool:
        print(f'Moving down {distance} cm')
        return True
    
    def rotate_cw(self, angle) -> bool:
        print(f'Rotating clockwise {angle} degrees')
        return True
    
    def rotate_ccw(self, angle) -> bool:
        print(f'Rotating counter-clockwise {angle} degrees')
        return True
    
    
    
if __name__=="__main__":
    drone=VirtualDrone()
    print(drone.get_all_funcs_desc())