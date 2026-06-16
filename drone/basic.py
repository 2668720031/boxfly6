from abc import ABC, abstractmethod
import json
class BasicDrone(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def takeoff(self) -> bool:
        '''          
        {
            "name": "takeoff",
            "description": "Takeoff the drone",
            "params": {},
            "return": "bool",
            "examples": ["drone.takeoff()"]
        }
        '''
        pass

    @abstractmethod
    def land(self) -> bool:
        '''
        {
            "name": "land",
            "description": "Land the drone",
            "params": {},
            "return": "bool",
            "examples": ["drone.land()"]
        }
        '''
        pass

    @abstractmethod
    def move_forward(self, distance) -> bool:
        '''
        {
            "name": "move_forward",
            "description": "Move the drone forward. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.move_forward(50)", "drone.move_forward(val=10)"]
        }
        '''
        pass

    @abstractmethod
    def move_backward(self, distance) -> bool:
        '''
        {
            "name": "move_backward",
            "description": "Move the drone backward. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.move_backward(50)", "drone.move_backward(distance=100)"]
        }
        '''
        pass

    @abstractmethod
    def move_left(self, distance) -> bool:
        '''
        {
            "name": "move_left",
            "description": "Move the drone left. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.move_left(50)", "drone.move_left(distance=100)"]
        }
        '''
        pass

    @abstractmethod
    def move_right(self, distance) -> bool:
        '''
        {
            "name": "move_right",
            "description": "Move the drone right. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.move_right(50)", "drone.move_right(distance=100)"]
        }
        '''
        pass

    @abstractmethod
    def move_up(self, distance) -> bool:
        '''
        {
            "name": "move_up",
            "description": "Move the drone up. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.move_up(50)", "drone.move_up(distance=100)"]
        }
        '''
        
        pass

    @abstractmethod
    def move_down(self, distance) -> bool:
        '''
        {
            "name": "move_down",
            "description": "Move the drone down. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.move_down(50)", "drone.move_down(distance=100)"]
        }
        '''
        pass

    @abstractmethod
    def rotate_cw(self, angle) -> bool:
        '''
        {
            "name": "rotate_cw",
            "description": "Rotate the drone clockwise. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.rotate_cw(90)", "drone.rotate_cw(angle=180)"]
        }
        '''
        pass

    @abstractmethod
    def rotate_ccw(self, angle) -> bool:
        '''
        {
            "name": "rotate_ccw",
            "description": "Rotate the drone counter clockwise. Each time the function is called, the joystick will be pushed with a certain force for one second.",
            "params": {"val": "int, 0-100, joystick force"},
            "return": "bool",
            "examples": ["drone.rotate_ccw(90)", "drone.rotate_ccw(angle=180)"]
        }
        '''
        pass

    def get_func_desc(self, func:str):
        try:
            doc=getattr(BasicDrone, func).__doc__
        except:
            return {}
        if doc:
            return json.loads(doc)
        return {}
    
    def get_all_funcs_desc(self):
        # return {func: self.get_func_desc(func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}
        all_funcs_desc_list=[]
        for func in dir(self):
            if callable(getattr(self, func)) and not func.startswith("__"):
                # print(func)
                desc=self.get_func_desc(func)
                if desc!={}:
                    all_funcs_desc_list.append(desc)
                # print()
        return all_funcs_desc_list
    

