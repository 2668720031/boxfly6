import datetime
from rich import print
def print_t(*args, **kwargs):
    # Get the current timestamp
    current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Use built-in print to display the timestamp followed by the message
    print(f"[{current_time}]", *args, **kwargs)
    

import re

def extract_code_blocks(markdown_text):
    """
    从Markdown文本中提取所有代码块
    返回一个列表，每个元素是一个字典，包含语言和代码内容
    """
    pattern = r'```(\w*)\n([\s\S]*?)\n```'
    matches = re.findall(pattern, markdown_text)
    return [{'language': match[0], 'code': match[1]} for match in matches]

if __name__=='__main__':
    text="""
```python\ndrone.move_forward(50)\n```
"""
    print(extract_code_blocks(text))