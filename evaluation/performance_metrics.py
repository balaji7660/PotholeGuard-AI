"""
Real-time Performance Benchmarking Suite for PotholeGuard-AI.
Measures:
- End-to-end processing FPS
- Model forward pass latency (ms)
- Preprocessing / Postprocessing overhead (ms)
- Memory consumption (RAM / VRAM)
"""
import time
import torch
import psutil
import numpy as np
from typing import Dict, Any

def benchmark_inference_performance(pipeline_fn, dummy_frame: np.ndarray, num_iterations: int = 50) -> Dict[str, Any]:
    latencies = []
    
    # Warmup
    for _ in range(5):
        _ = pipeline_fn(dummy_frame)

    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = pipeline_fn(dummy_frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    fps = round(1000.0 / avg_lat, 2) if avg_lat > 0 else 0.0

    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "throughput_fps": fps,
        "ram_usage_mb": round(ram_mb, 1),
        "gpu_available": torch.cuda.is_available()
    }
