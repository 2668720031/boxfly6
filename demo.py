import pygame
import cv2
import numpy as np
import gradio as gr

import io
import base64

from time import sleep, time
from threading import Thread
from PIL import Image
from drone import TelloPy
from prompt import get_message_template
from utils import extract_code_blocks
from rich import print
from openai import OpenAI

import requests
import random

client = OpenAI(
    api_key='EMPTY',
    base_url='http://10.113.163.111:8889/v1',
)


answer=[
"""
```python
drone.move_left(35)
```

Explanation: Find a drone in the image in the left part of the image and move left to keep it in the center of the image.

Description: The image shows an indoor setting with a drone in the left part of the image. The drone has four propellers. The lighting is bright, and the image is clear. I will move left to keep the drone in the center of the image.
""",
"""
```python
drone.move_right(30)
```

Explanation: Find a drone in the image in the right part of the image. According to the user's instruction, I will move right to keep it in the center of the image.

Description: The image shows an indoor setting with a drone in the right part of the image, likely in a office or laboratory environment. I will move right to keep the drone in the center of the image.
""",
"""
```python
drone.move_up(30)
```

Explanation: A drone is in the top part of the image. Move up to keep it in the center of the image.
Description: The image shows a drone in the top part of the image. The drone is flying in the air. I will move up to keep the drone in the center of the image.
""",
"""
```python
drone.move_down(30)
```

Explanation: Searching for the drone in the image. The drone is in the bottom part of the image. Move down to keep it in the center of the image.

Description: The image shows a drone in the bottom part of the image. The drone is flying in the air. Move the camera down to keep the drone in the center of the image.
"""
]

current_action=0

def ask(messages):
    global current_action
    
    start_time=time()
    
    sleep(2+random.random())

    output_text=answer[current_action]
    
    print(output_text)
    
    current_action+=1
    
    end_time=time()
    
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": output_text,
                }
            ]
        }
    )
    
    print(f"Time taken: {end_time-start_time:.2f}s")
    return output_text, messages

messages=None

def get_last_assisant_message():
    global messages
    if messages is None:
        return 'No command yet'
    res=[]
    for message in messages[::-1]:
        if message['role']=='assistant':
            res.append(message['content'][0]['text'])
    if len(res)>0:
        for i in range(len(res)):
            res[i]= f'### Step {i+1}\n'+res[i]
        
        return "\n\n".join(res[::-1])

    return 'No command yet'

def delete_image(messages):
    for message in messages:
        if message['content'][0]['type']=='image' or message['content'][0]['type']=='image_url':
            message['content'].pop(0)
            if message['role']=='system':
                message['content'][0]['text']="Continue to generate the next step."
    return messages


def process_instruction(instruction):
    last_image= Image.fromarray(cv2.cvtColor(drone.current_image, cv2.COLOR_BGR2RGB))

    buf=io.BytesIO()
    last_image.save(buf, format='JPEG')
    last_image_str=base64.b64encode(buf.getvalue()).decode('utf-8')
    
    global messages
    
    messages = get_message_template()

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url":f"data:image;base64,{last_image_str}",
                    },
            },
            {
                "type": "text",
                "text": instruction,
            }
        ]
    })
    
    print('Processing instruction:', instruction)
    
    is_first_message=True
    
    while True:
        output_text, messages = ask(messages)
        
        print('Assistant:', output_text)
        
        if 'break' in output_text:
            break
        
        res=extract_code_blocks(output_text)
        
        # drone.set_zero()
        if len(res)>0:
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
            break
        
        messages=delete_image(messages)
        
        if is_first_message:
            sleep(1)
            is_first_message=False
        
        last_image= Image.fromarray(cv2.cvtColor(drone.current_image, cv2.COLOR_BGR2RGB))
        buf=io.BytesIO()
        last_image.save(buf, format='JPEG')
        last_image_str=base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # messages = get_message_template()

        # messages.append({
        #     "role": "user",
        #     "content": [
        #         {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url":f"data:image;base64,{last_image_str}",
        #                 },
        #         },
        #         {
        #             "type": "text",
        #             "text": instruction,
        #         }
        #     ]
        # })
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":f"data:image;base64,{last_image_str}",
                        },
                    },
                    {
                        "type": "text",
                        "text": """The image is captured by the drone after executing the last command. Generate the next step. If you think the instruction is finished, return 'break' in code block.""",
                    }
                ],
            }
        )
        
drone = TelloPy()

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


drone.connect()
drone.takeoff()

with gr.Blocks() as ui:
    gr.Interface(
            process_instruction,
            'textbox', 
            None
        )
    gr.Textbox(
        label="Assistant Message List",
        value=get_last_assisant_message,
        every=0.01,
    )

ui.launch()

drone.land()
drone.quit()
