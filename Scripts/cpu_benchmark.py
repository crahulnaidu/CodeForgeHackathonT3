# cpu_benchmark.py
# Benchmarks CPU inference latency for baseline vs CBAM-enhanced YOLOv8 models.

import time
import torch
from ultralytics import YOLO

IMAGE = "test6.png"            # change if needed
BASE_MODEL = "baseline_trained.pt"
CBAM_MODEL = "cbam_stage2.pt"

RUNS = 50   # number of forward passes to average over


def benchmark(model_path):
    """Measure average inference time on CPU over multiple runs."""
    print("\nLoading:", model_path)

    model = YOLO(model_path)

    # Force CPU execution
    device = "cpu"

    # Warmup passes to stabilize timings
    for _ in range(5):
        model.predict(IMAGE, device=device, imgsz=960, conf=0.25, verbose=False)

    start = time.time()

    for _ in range(RUNS):
        model.predict(IMAGE, device=device, imgsz=960, conf=0.25, verbose=False)

    end = time.time()

    avg = (end - start) / RUNS

    print("Average inference time:", avg, "seconds")
    print("FPS:", 1 / avg)

    return avg


if __name__ == "__main__":

    base_time = benchmark(BASE_MODEL)
    cbam_time = benchmark(CBAM_MODEL)

    print("\n===== CPU COMPARISON =====")
    print(f"Baseline time : {base_time:.4f} sec")
    print(f"CBAM time     : {cbam_time:.4f} sec")

    diff = ((cbam_time - base_time) / base_time) * 100
    print(f"Latency increase: {diff:.2f}%")