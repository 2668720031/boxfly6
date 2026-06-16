
from rich import print
import time
import cv2

from PIL import Image
import io

from prompt import get_message_template
import os
import openai

import base64

client=openai.OpenAI(
    base_url='http://10.113.163.111:8000/v1',
)

start_time=time.time()
image_path="/home/box/ssd/BoxFly/image/test.png"

image = cv2.imread(image_path)
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
buf=io.BytesIO()
image.save(buf, format='JPEG')
image_str=base64.b64encode(buf.getvalue()).decode('utf-8')
    
messages = get_message_template()

messages=[
   {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "If you see TWO people in the image, move forward, otherwise if you see ONE person, move right."},
            {
                "type": "image_url",
                "image": f"data:image/jpeg;base64,{image_str}",
            },
        ],
    } 
]
response = client.chat.completions.create(
    model='Qwen/Qwen2.5-VL-7B-Instruct',
    messages=messages,
)
print(f"Time: {time.time()-start_time}")
print(response)