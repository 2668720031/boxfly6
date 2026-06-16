import base64
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = 'http://10.113.163.111:8889/v1'
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
image_path = "./images.jpeg"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_qwen = f"data:image;base64,{encoded_image_text}"
chat_response = client.chat.completions.create(
    model="./Qwen2.5-VL-7B-Instruct/",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_qwen
                    },
                },
                {"type": "text", "text": "Describe the image."},
            ],
        },
    ],
)
print("Chat response:", type(chat_response), chat_response)