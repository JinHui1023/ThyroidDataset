# benchmark.py
import time
import psutil
import numpy as np
from trt_inference import TRTInference

def benchmark_inference(engine_path, image_path, num_runs=100):
    """基准测试推理性能和能效"""
    trt_engine = TRTInference(engine_path)
    
    # 预热
    _, _ = trt_engine.infer(image_path)
    
    # 性能测试
    start_time = time.time()
    for _ in range(num_runs):
        _, _ = trt_engine.infer(image_path)
    total_time = time.time() - start_time
    
    # 计算指标
    avg_latency = total_time / num_runs * 1000  # 毫秒
    fps = num_runs / total_time
    
    # 能耗测量
    process = psutil.Process()
    energy_usage = process.cpu_percent() * total_time / 100  # 近似能耗
    
    print(f"平均延迟: {avg_latency:.2f} ms")
    print(f"帧率: {fps:.2f} FPS")
    print(f"每次推理能耗: {energy_usage / num_runs * 1000:.2f} mJ")
    print(f"总推理时间: {total_time:.2f} 秒 ({num_runs} 次推理)")

if __name__ == "__main__":
    benchmark_inference("thyroid_model.trt", "test_image.png")
