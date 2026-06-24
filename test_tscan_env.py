import torch
import numpy as np
import time

try:
    from TS_CAN import TSCAN
    import_success = True
except Exception as e:
    import_success = False
    import_error = e

def verify_environment_and_model():
    print("\n" + "="*45)
    print("🚀 TS-CAN 核心环境与底层推理兼容性最终检测 🚀")
    print("="*45)

    # ------------------ 阶段一：基础库检测 ------------------
    print("\n[阶段一：基础运行库检测]")
    print(f"  [-] PyTorch 版本 : {torch.__version__}")
    print(f"  [-] Numpy 版本   : {np.__version__}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [-] 计算硬件分配 : {device}")
    
    if not import_success:
        print(f"\n❌ 致命错误：TS_CAN 模块导入失败。原因: {import_error}")
        return
    else:
        print("  [✅] TS_CAN 模块代码导入成功！")

    # ------------------ 阶段二：模型推理检测 ------------------
    print("\n[阶段二：模型前向推理 (Forward Pass) 检测]")
    try:
        # 1. 实例化模型
        print("  [-] 正在将 TS-CAN 模型加载到显存...")
        model = TSCAN(frame_depth=10).to(device)
        model.eval()
        print("  [✅] 模型实例化成功！")

        # 2. 构造已知正确的 6 通道输入张量
        # 形状：[时间帧数=10, 组合通道数=6, 宽=36, 高=36]
        print("  [-] 正在构建匹配源码的 [10, 6, 36, 36] 张量...")
        dummy_input = torch.randn(10, 6, 36, 36, dtype=torch.float32).to(device)
        
        # 3. 压测与推理
        print("  [-] 正在执行前向推理测试...")
        
        # 预热一下 GPU (消除首次初始化的耗时)
        with torch.no_grad():
            _ = model(dummy_input)
            
        # 正式计时测速
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        end_time = time.time()
        
        print("  [✅] 前向推理完美通过！没有 unpack 报错！")
        print(f"  [✅] 模型输出形状 : {output.shape}")
        print(f"  [✅] 单次推理耗时 : {(end_time - start_time) * 1000:.2f} 毫秒")

        print("\n🎉 结论：环境极度健康！你的 Orin AGX 随时可以开始处理无人机视频流！\n")

    except Exception as e:
        print(f"\n❌ 推理阶段崩溃，请检查错误日志: {e}")

if __name__ == "__main__":
    verify_environment_and_model()