import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from evaluation.segmentation_metrics import compute_segmentation_metrics
from evaluation.depth_metrics import compute_depth_metrics
from evaluation.risk_metrics import compute_risk_metrics
from evaluation.tracking_metrics import compute_tracking_metrics
from evaluation.performance_metrics import benchmark_inference_performance
from evaluation.ablation_study import run_ablation_comparison
from evaluation.robustness_benchmark import run_robustness_benchmark
from inference.pipeline import PotholeGuardPipeline

def run_all_evaluations():
    print("================================================================")
    print("         [*] RUNNING POTHOLEGUARD-AI EVALUATION SUITE           ")
    print("================================================================")

    # 1. Segmentation Synthetic Test
    gt_mask = np.zeros((512, 512), dtype=np.uint8)
    pred_mask = np.zeros((512, 512), dtype=np.uint8)
    gt_mask[100:200, 100:200] = 255
    pred_mask[105:205, 105:205] = 255
    seg_res = compute_segmentation_metrics(pred_mask, gt_mask)
    print(f"[+] Segmentation Metrics Test: Dice = {seg_res['dice']}, IoU = {seg_res['iou']}")

    # 2. Depth Metrics Test
    gt_depth = np.ones((512, 512), dtype=np.float32) * 0.8
    pred_depth = np.ones((512, 512), dtype=np.float32) * 0.78
    depth_res = compute_depth_metrics(pred_depth, gt_depth)
    print(f"[+] Depth Metrics Test: RMSE = {depth_res['rmse']}, MAE = {depth_res['mae']}")

    # 3. Risk Engine Metrics Test
    y_true = ["SAFE", "WARNING", "DANGER", "DANGER", "UNCERTAIN"]
    y_pred = ["SAFE", "WARNING", "DANGER", "DANGER", "UNCERTAIN"]
    risk_res = compute_risk_metrics(y_true, y_pred)
    print(f"[+] Risk Engine Accuracy: {risk_res['overall_accuracy'] * 100}%")

    # 4. Tracking Metrics Test
    trk_res = compute_tracking_metrics(10, 10, 0, 10, 45)
    print(f"[+] Duplicate Warning Suppression Efficiency: {trk_res['suppression_efficiency'] * 100}%")

    # 5. Performance Benchmark
    pipe = PotholeGuardPipeline()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    perf_res = benchmark_inference_performance(pipe.process_frame, dummy, num_iterations=20)
    print(f"[+] Processing Throughput: {perf_res['throughput_fps']} FPS (Mean Latency: {perf_res['mean_latency_ms']} ms)")

    # 6. Ablation and Robustness Studies
    run_ablation_comparison()
    run_robustness_benchmark()
    print("[+] Ablation & Robustness benchmarks generated in experiments/")
    print("================================================================")

if __name__ == "__main__":
    run_all_evaluations()
