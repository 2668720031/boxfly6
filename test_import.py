import sys

print("=========================================")
print("当前使用的 Python 解释器路径:", sys.executable)
print("当前使用的 Python 版本:", sys.version.split(' ')[0])
print("当前 sys.path 搜索路径:")
for p in sys.path:
    print(" -", p)
print("=========================================\n")

# 测试 1：检查是否能找到 tellopy 包
print(">>> 测试 1: 导入 tellopy 基础包")
try:
    import tellopy
    print("[成功] 成功导入 tellopy!")
    print("       实际加载的 tellopy 路径是:", tellopy.__file__)
except ImportError as e:
    print(f"[失败] 无法导入 tellopy。报错: {e}")
    print("       请检查你是否在正确的 conda 虚拟环境中运行。")
    sys.exit(1)

print("\n>>> 测试 2: 尝试直接导入 LogData (你报错的那行代码)")
try:
    from tellopy._internal.protocol import LogData
    print("[成功] 成功直接导入 LogData 类!")
    print("       类信息:", LogData)
except ImportError as e:
    print(f"[失败] 直接导入失败。报错: {e}")

print("\n>>> 测试 3: 尝试导入 protocol 模块本身并检查内容")
try:
    from tellopy._internal import protocol
    print("[成功] 成功导入 protocol 模块!")
    
    # 检查模块里面到底有没有 LogData
    if hasattr(protocol, 'LogData'):
        print("[成功] protocol 模块内部确实存在 'LogData' 类!")
    else:
        print("[失败] protocol 模块被找到了，但里面没有名为 'LogData' 的类。")
        print("       当前 protocol 模块里可用的类和函数有:")
        available_items = [item for item in dir(protocol) if not item.startswith('__')]
        print("       ", available_items)
except ImportError as e:
    print(f"[失败] 连 protocol 模块都无法导入。报错: {e}")
print("\n=========================================")