"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2:
- label smoothing (trang 56)
- focal loss (trang 57)
- class-balanced loss (trang 57)
- Mixup/CutMix (trang 48)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, target)
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).

    Khi gamma = 0 và alpha = None, thu về đúng cross-entropy thông thường.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log_p: (N, C), p: (N, C)
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # Trích xuất p_t và log(p_t) của nhãn đúng
        target = target.view(-1, 1)
        log_pt = log_p.gather(1, target).squeeze(1)
        pt = p.gather(1, target).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, target.squeeze(1))
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: 'ce', 'ls', 'focal', 'ce_weighted'."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loại loss: {kind}")


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1.
    - beta > 0: class-balanced theo số mẫu hiệu dụng (Cui et al. arXiv:1901.05555):
        w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng bằng NUM_CLASSES.
    """
    counts = np.asarray(counts, dtype=np.float64)
    k = len(counts)

    if beta <= 0.0:
        inv = 1.0 / np.maximum(counts, 1.0)
        weights = inv / inv.mean()
    else:
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * k

    return torch.tensor(weights, dtype=torch.float32)


def _rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    """Tạo toạ độ hộp cắt chữ nhật cho CutMix."""
    _, _, h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix)."""
    if alpha <= 0.0:
        return x, (y, y, 1.0)

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)
    elif mode == "cutmix":
        x_mixed = x.clone()
        bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
        # Điều chỉnh lam theo diện tích thực tế của hộp cắt sau khi bị cắt ở biên
        box_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = x.size(2) * x.size(3)
        lam_adjusted = 1.0 - (box_area / float(total_area))
        return x_mixed, (y_a, y_b, lam_adjusted)
    else:
        return x, (y, y, 1.0)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính loss kết hợp cho batch đã trộn: lam * loss(y_a) + (1 - lam) * loss(y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
