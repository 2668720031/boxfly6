import cv2
import threading
import time
import os
from datetime import datetime

class VideoRecorder:
    def __init__(self, src="udp://@127.0.0.1:5000", output_dir="./records"):
        # 视频源设置
        self.cap = cv2.VideoCapture(src)
        
        # 初始视频参数
        self.frame_size = None
        self.fps = None
        
        # 录制控制相关
        self.recording = False
        self.output_path = ""
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # 路径生成函数（默认使用时间戳）
        self.path_generator = lambda: os.path.join(
            self.output_dir, 
            f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        )
        
        # 视频写入器
        self.writer = None
        
        # 启动读取线程
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _generate_output_path(self):
        """生成新的输出路径"""
        os.makedirs(self.output_dir, exist_ok=True)
        return self.path_generator()

    def _update(self):
        """主循环：读取视频帧"""
        while True:
            # 读取视频帧
            ret, frame = self.cap.read()
            if self.frame_size is None:
                self.frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))
            if self.fps is None:
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            print(f"Frame size: {self.frame_size}, FPS: {self.fps}")
            if not ret:
                break
            
            with self.lock:
                # 录制处理
                if self.recording:
                    if self.writer is None:
                        codec = cv2.VideoWriter_fourcc(*'XVID')
                        self.writer = cv2.VideoWriter(
                            self.output_path,
                            codec,
                            self.fps,
                            self.frame_size
                        )
                    self.writer.write(frame)

            time.sleep(1/self.fps)  # 近似实时速度

    def start_recording(self):
        """开始录制"""
        with self.lock:
            if not self.recording:
                self.output_path = self._generate_output_path()
                self.recording = True

    def stop_recording(self):
        """停止录制"""
        with self.lock:
            if self.recording:
                self.recording = False
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None

    def set_path_generator(self, generator_func):
        """自定义路径生成函数"""
        with self.lock:
            self.path_generator = generator_func

    def release(self):
        """释放资源"""
        self.stop_event.set()
        self.stop_recording()
        self.cap.release()
        if self.thread.is_alive():
            self.thread.join()

# 使用示例
if __name__ == "__main__":
    # 初始化视频录制器（默认使用摄像头）
    recorder = VideoRecorder()
    
    # 示例控制逻辑
    try:
        # 第一次录制
        recorder.start_recording()
        print("开始第一次录制...")
        time.sleep(5)
        recorder.stop_recording()
        print(f"视频保存至：{recorder.output_path}")
        
        # 第二次录制（自定义路径）
        recorder.set_path_generator(lambda: "./records/custom_name.avi")
        recorder.start_recording()
        print("开始第二次录制...")
        time.sleep(3)
        recorder.stop_recording()
        print(f"视频保存至：{recorder.output_path}")
        
    finally:
        recorder.release()