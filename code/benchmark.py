"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau mỗi lượt đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, phiên bản PyTorch
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây (ms).

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo đạc chính xác
    durations_ms = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()

        fn()

        if sync is not None:
            sync()
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    durations = np.array(durations_ms, dtype=np.float64)
    p50 = float(np.percentile(durations, 50))
    p95 = float(np.percentile(durations, 95))
    p99 = float(np.percentile(durations, 99))
    mean = float(np.mean(durations))

    return {
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "p99": round(p99, 2),
        "mean": round(mean, 2),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    tensor_dtype = torch.float32
    if dtype == "fp16":
        tensor_dtype = torch.float16
        model = model.half()

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device, dtype=tensor_dtype)

    def forward_fn():
        with torch.inference_mode():
            if dtype == "amp" and target_device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    model(dummy_input)
            else:
                model(dummy_input)

    sync_fn = torch.cuda.synchronize if target_device.type == "cuda" else None
    results = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if (target_device.type == "cuda" and torch.cuda.is_available()) else "CPU"
    throughput = round(batch_size / (results["p50"] / 1000.0), 1) if results["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": results["p50"],
        "p95": results["p95"],
        "p99": results["p99"],
        "images_per_s": throughput,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ của TTA K views (slide trang 63: chi phí gần tuyến tính theo K)."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    dummy_views = [
        torch.randn(batch_size, 3, img_size, img_size, device=target_device, dtype=torch.float32)
        for _ in range(k_views)
    ]

    def forward_tta():
        with torch.inference_mode():
            for v in dummy_views:
                model(v)

    sync_fn = torch.cuda.synchronize if target_device.type == "cuda" else None
    results = bench(forward_tta, warmup=warmup, iters=iters, sync=sync_fn)
    return results
