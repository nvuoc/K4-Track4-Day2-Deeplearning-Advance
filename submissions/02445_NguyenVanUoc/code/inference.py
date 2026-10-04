"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Bao gồm:
- TTA lật ngang (I01)
- TTA multi-crop (I02)
- Gộp xác suất vs logit (I03)
- Dò độ phân giải (I04)
- Ensemble nhiều model (I05)
- Temperature scaling & ECE (I07)
- Gộp Conv-BatchNorm (I08)
"""
from __future__ import annotations

import copy
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


def _softmax(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def predict_logits(model: nn.Module, loader, device, view=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for images, targets, fnames in loader:
            images = images.to(device)
            if view is not None:
                images = view(images)

            outputs = model(images)
            all_filenames.extend(fnames)
            all_y_true.append(targets.numpy())
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch ảnh (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> list[torch.Tensor]:
    """5-crop: 4 góc và trung tâm."""
    _, _, h, w = x.shape
    if h < crop or w < crop:
        # Nếu ảnh nhỏ hơn kích thước crop, resize về crop
        resized = F.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False)
        return [resized]

    top_left = x[:, :, :crop, :crop]
    top_right = x[:, :, :crop, w - crop:]
    bottom_left = x[:, :, h - crop:, :crop]
    bottom_right = x[:, :, h - crop:, w - crop:]
    cy, cx = (h - crop) // 2, (w - crop) // 2
    center = x[:, :, cy:cy + crop, cx:cx + crop]

    return [top_left, top_right, bottom_left, bottom_right, center]


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

    - space="prob":  trung bình softmax của từng view
    - space="logit": trung bình logit rồi softmax
    """
    if space == "prob":
        probs_list = [_softmax(z) for z in logits_per_view]
        avg_probs = np.mean(probs_list, axis=0)
        # Chuẩn hoá đảm bảo tổng = 1
        return avg_probs / np.sum(avg_probs, axis=-1, keepdims=True)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return _softmax(avg_logits)
    else:
        raise ValueError(f"Không hỗ trợ space: {space}")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    avg_probs = np.mean(list_of_probs, axis=0)
    return avg_probs / np.sum(avg_probs, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69)."""
    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    def nll_obj(T_val):
        T = float(T_val)
        if T <= 0.001:
            return 1e9
        scaled_logits = val_logits / T
        z_max = np.max(scaled_logits, axis=-1, keepdims=True)
        log_sum_exp = z_max + np.log(np.sum(np.exp(scaled_logits - z_max), axis=-1, keepdims=True))
        log_probs = scaled_logits - log_sum_exp
        nll = -log_probs[np.arange(len(val_labels)), val_labels].mean()
        return nll

    # Tối ưu hoá 1 chiều tìm T trong khoảng [0.1, 5.0]
    res = scipy.optimize.minimize_scalar(nll_obj, bounds=(0.1, 5.0), method="bounded")
    best_t = float(res.x)
    return round(best_t, 4)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    scaled = np.asarray(logits, dtype=np.float64) / float(T)
    return _softmax(scaled)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm2d vào Conv2d liền trước khi suy luận (slide trang 71, 75).

    w' = gamma * w / sqrt(var + eps)
    b' = beta + gamma * (b - mean) / sqrt(var + eps)
    """
    model = copy.deepcopy(model)
    model.eval()

    def _fuse_module(m: nn.Module):
        children = list(m.named_children())
        for i in range(len(children)):
            name_curr, child_curr = children[i]
            _fuse_module(child_curr)

            if i + 1 < len(children):
                name_next, child_next = children[i + 1]
                if isinstance(child_curr, nn.Conv2d) and isinstance(child_next, nn.BatchNorm2d):
                    # Tiến hành fuse Conv2d + BatchNorm2d
                    conv = child_curr
                    bn = child_next

                    fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                    setattr(m, name_curr, fused_conv)
                    setattr(m, name_next, nn.Identity())

    _fuse_module(model)
    return model
