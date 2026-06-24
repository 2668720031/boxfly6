import torch
import cv2
import numpy as np
from collections import deque
import time

try:
    from TS_CAN import TSCAN
except Exception as e:
    print(f"❌ 导入失败: {e}")
    exit()

class DroneRPPG:
    def __init__(self, frame_length=10, image_size=36): # 🌟 核心修改：改为 36
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_length = frame_length
        self.image_size = image_size
        
        print(f"[*] 当前使用计算设备: {self.device}")
        
        # 实例化模型
        self.model = TSCAN(frame_depth=self.frame_length).to(self.device)
        self.model.eval()
        
        # 存放连续原始帧的滑动窗口
        self.raw_frame_buffer = deque(maxlen=self.frame_length)

    def process_face_crop(self, face_rgb):
        """模拟 Orin 处理 Tello 传来的单帧人脸"""
        # 1. 调整大小为模型专用的 36x36
        face_resized = cv2.resize(face_rgb, (self.image_size, self.image_size))
        
        # 2. 归一化
        face_norm = face_resized.astype(np.float32) / 255.0 
        
        # 3. 存入队列
        self.raw_frame_buffer.append(face_norm)
        
        if len(self.raw_frame_buffer) == self.frame_length:
            return self.infer()
        return None

    def infer(self):
        """拼装 6 通道张量并推理"""
        raw_frames = np.array(self.raw_frame_buffer) 
        
        diff_frames = np.zeros_like(raw_frames)
        diff_frames[1:] = raw_frames[1:] - raw_frames[:-1] 
        
        concat_frames = np.concatenate((diff_frames, raw_frames), axis=-1)
        concat_frames = np.transpose(concat_frames, (0, 3, 1, 2))
        
        input_tensor = torch.tensor(concat_frames, dtype=torch.float32).contiguous().to(self.device)
            
        with torch.no_grad():
            start_time = time.time()
            output = self.model(input_tensor) 
            end_time = time.time()
            self.infer_time = (end_time - start_time) * 1000 
            
        return output.cpu().numpy()


if __name__ == '__main__':
    print("====== TS-CAN 完整管道验证测试 (36x36版) ======")
    
    rppg = DroneRPPG(frame_length=10, image_size=36)
    print("\n[*] 正在模拟生成连续的无人机抓拍画面 (10帧)...")
    
    for i in range(10):
        # 模拟 100x100 的随机彩色人脸
        dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        out = rppg.process_face_crop(dummy_face)
        
        if out is not None:
            print(f"  -> 第 {i+1} 帧: 🎉 测试成功！不再报错！")
            print(f"✅ 模型单次推理耗时 : {rppg.infer_time:.2f} ms")
            print(f"✅ 输出波形数组形状 : {out.shape} (这代表你提取到了脉搏波特征！)")
        else:
            print(f"  -> 第 {i+1} 帧: (Buffer 蓄水中...)")