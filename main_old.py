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


client = OpenAI(
    api_key='EMPTY',
    base_url='http://10.113.182.9:8889/v1/',
)

# client = OpenAI(
#     api_key='EMPTY',
#     base_url='https://vllm.boxz.dev/v1',
# )

# client = OpenAI(
#     api_key='EMPTY',
#     base_url='http://10.113.178.170:8889/v1',
# )

# client = OpenAI(
#     api_key='sk-oisTMKwNduAkIFj8Ea1fA7EcDf8144A39dCf9e0d1261A3D8',
#     base_url='https://www.gptapi.us/v1',
# )

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# model_path="/home/box/ssd/Qwen2.5-VL-7B-Instruct/"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path, torch_dtype='auto', device_map="auto"
# )
# model.eval()

# processor = AutoProcessor.from_pretrained(model_path)

# @torch.no_grad()
# def ask(messages):
#     start_time=time()
    
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)

#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")

#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=1024)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
    
#     messages.append(
#         {
#             "role": "assistant",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": output_text[0],
#                 }
#             ]
#         }
#     )
    
#     end_time=time()
    
#     print(f"Time taken: {end_time-start_time:.2f}s")
#     return output_text[0], messages

def ask(messages):
    start_time=time()
    
    response = client.chat.completions.create(
        model='Qwen2.5-VL-7B-Instruct',
        # model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        messages=messages,
        # response_format={"type": "json_object"},
    )

    # response = requests.post(
    #     headers={
    #         "Content-Type": "application/json",
    #     },
    #     url='http://10.113.163.111:8889/v1/chat/completions',
    #     data={
    #         "model": './Qwen2.5-VL-7B-Instruct/',
    #         "messages": messages,
    #     }
    # )

    # response = response.json()
    # print(response)
    # response=response.json()
    # print(type(response), response)
    print(response.choices[0].message.content)
    output_text=response.choices[0].message.content
    
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
    
    end_time=time()
    
    print(f"Time taken: {end_time-start_time:.2f}s")
    return output_text, messages

messages=None

def get_last_assisant_message():
    global messages
    if messages is None:
        return 'No command yet'
    res=[]
    for message in messages:
        if message['role']=='assistant':
            res.append(message['content'][0]['text'])
    if len(res)>0:
        for i in range(len(res)):
            res[i]= f'Step {i+1}\n'+res[i]
        
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
                        # "text": """The image is captured by the drone after executing the last command. Generate the next step. If you think the instruction is finished, return 'break' in code block.""",
                        "text": """The image is captured by the drone after executing the last command. Generate the next step.""",
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
