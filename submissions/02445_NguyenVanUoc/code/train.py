"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Đảm bảo import được dataset, model, losses, eval
CURR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURR_DIR.parent
if str(CURR_DIR) not in sys.path:
    sys.path.insert(0, str(CURR_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dataset as ds_module
import model as model_module
import losses as losses_module
import eval as ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (S4) ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = model_module.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-4, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler: GradScaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    # Nếu backbone bị đóng băng, BatchNorm của backbone phải ở chế độ eval
    if cfg.init == "frozen" or getattr(model, "backbone_frozen", False):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    total_loss = 0.0
    count = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if cfg.mix:
            images, mixed_targets = losses_module.mix_batch(
                images, targets, alpha=cfg.mix_alpha, mode=cfg.mix
            )

        optimizer.zero_grad(set_to_none=True)

        use_amp = cfg.amp and (device.type == "cuda")
        with autocast(enabled=use_amp):
            outputs = model(images)
            if cfg.mix:
                loss = losses_module.mixed_loss(criterion, outputs, mixed_targets)
            else:
                loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * len(targets)
        count += len(targets)

    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": total_loss / max(1, count), "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_fnames = []
    all_y_true = []
    all_logits = []
    total_loss = 0.0
    count = 0

    with torch.inference_mode():
        for images, targets, fnames in loader:
            images = images.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets_dev)

            total_loss += loss.item() * len(targets)
            count += len(targets)

            all_fnames.extend(fnames)
            all_y_true.append(targets.numpy())
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, count)

    # Tính xác suất softmax
    z_max = np.max(logits, axis=-1, keepdims=True)
    exp_z = np.exp(logits - z_max)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    # Sử dụng eval.py để tính chỉ số chuẩn của lớp
    cm = ev.confusion_matrix(y_true, probs.argmax(1), k=ev.NUM_CLASSES)
    metrics = ev.compute_metrics(probs, y_true, cm)
    metrics["loss"] = avg_loss

    return all_fnames, y_true, logits, avg_loss, metrics


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    val_acc = [h["val_top1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Subplot 1: Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o", color="#2563eb")
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="s", color="#dc2626")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Metrics
    axes[1].plot(epochs, val_f1, label="Val Macro-F1", marker="^", color="#16a34a", linewidth=2)
    axes[1].plot(epochs, val_acc, label="Val Top-1 Acc", marker="d", color="#9333ea", linestyle="--")
    best_idx = np.argmax(val_f1)
    axes[1].scatter([epochs[best_idx]], [val_f1[best_idx]], color="red", s=100, zorder=5, label=f"Best F1: {val_f1[best_idx]:.4f}")
    axes[1].set_title(f"{title} — Validation Performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    t_start = time.time()
    set_seed(cfg.seed)

    save_dir = run_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    # 1. Ghi config.json
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # 2. Đọc và kiểm tra dữ liệu
    train_df, val_df, test_df = ds_module.load_split(cfg.labels_dir, fold=cfg.fold)
    # Kiểm tra tính toàn vẹn (không chặn nếu images_dir chưa tồn tại trên máy dev cục bộ)
    ds_module.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = ds_module.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = ds_module.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds_module.make_loader(
        train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = ds_module.make_loader(
        val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.amp) else "cpu")

    # 3. Model, Loss, Optimizer, Scheduler, EMA
    model = model_module.build_model(
        cfg.backbone, pretrained=True, num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init
    )
    model.to(device)

    # Chuẩn bị criterion
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = np.bincount(train_df["Label"].to_numpy(dtype=np.int64), minlength=ev.NUM_CLASSES)
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        w = losses_module.class_weights(counts, beta=beta).to(device)
        criterion = losses_module.build_criterion("ce_weighted", weight=w)
    elif cfg.loss == "ls":
        criterion = losses_module.build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    elif cfg.loss == "focal":
        criterion = losses_module.build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        criterion = losses_module.build_criterion("ce")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay else None

    num_params = model_module.count_params(model)
    gmacs = model_module.count_gmacs(model, img_size=cfg.img_size)

    # 4. Vòng lặp huấn luyện
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_model_state = None
    epoch_times = []

    print(f"\n[{cfg.exp_id}] Bắt đầu huấn luyện {cfg.backbone} ({cfg.init}) | Seed: {cfg.seed} | Epochs: {cfg.epochs}")

    for epoch in range(1, cfg.epochs + 1):
        t_ep0 = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        t_ep = time.time() - t_ep0
        epoch_times.append(t_ep)

        # Đánh giá trên tập val
        if ema is not None:
            ema.apply_shadow(model)

        val_fnames, val_targets, val_logits, val_loss, val_m = evaluate(model, val_loader, criterion, device)

        if ema is not None:
            ema.restore(model)

        val_f1 = val_m["macro_f1"]
        val_top1 = val_m["top1_acc"]

        ep_log = {
            "epoch": epoch,
            "train_loss": round(train_res["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_top1": round(val_top1, 4),
            "lr": train_res["lr"],
            "time_s": round(t_ep, 2),
        }
        history.append(ep_log)
        print(f"  Epoch {epoch:02d}/{cfg.epochs:02d} | Train Loss: {train_res['train_loss']:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Top-1: {val_top1:.4f} ({t_ep:.1f}s)")

        # Chọn checkpoint theo Macro-F1 val (hòa lấy epoch sớm hơn)
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            if ema is not None:
                ema.apply_shadow(model)
            best_model_state = copy.deepcopy(model.state_dict())
            if ema is not None:
                ema.restore(model)

    # 5. Lưu checkpoint và log lịch sử
    torch.save(best_model_state, save_dir / "best_model.pt")
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / "history.csv", index=False)

    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"{cfg.exp_id} ({cfg.backbone})")

    # 6. Đánh giá lại checkpoint tốt nhất trên tập VAL và lưu file dự đoán val
    model.load_state_dict(best_model_state)
    val_fnames, val_targets, val_logits, val_loss, final_val_m = evaluate(model, val_loader, criterion, device)

    z_max = np.max(val_logits, axis=-1, keepdims=True)
    exp_z = np.exp(val_logits - z_max)
    val_probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    np.save(save_dir / "val_logits.npy", val_logits)
    ev.save_predictions(pred_path(cfg, "val"), val_fnames, val_targets, val_probs)

    # 7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): Đánh giá TEST đúng một lần duy nhất
    test_m = None
    if cfg.save_test_predictions:
        print(f"[{cfg.exp_id}] Đánh giá DUY NHẤT một lần trên TEST (Seed {cfg.seed})...")
        test_loader = ds_module.make_loader(
            test_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_fnames, test_targets, test_logits, test_loss, test_m = evaluate(model, test_loader, criterion, device)

        z_max_t = np.max(test_logits, axis=-1, keepdims=True)
        exp_z_t = np.exp(test_logits - z_max_t)
        test_probs = exp_z_t / np.sum(exp_z_t, axis=-1, keepdims=True)

        np.save(save_dir / "test_logits.npy", test_logits)
        ev.save_predictions(pred_path(cfg, "test"), test_fnames, test_targets, test_probs)

    t_total = time.time() - t_start
    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "num_params_m": num_params,
        "gmacs": gmacs,
        "best_epoch": best_epoch,
        "val_macro_f1": round(final_val_m["macro_f1"], 4),
        "val_top1": round(final_val_m["top1_acc"], 4),
        "val_ece": round(final_val_m["ece"], 4),
        "avg_epoch_time_s": round(float(np.mean(epoch_times)), 2),
        "total_time_s": round(t_total, 2),
        "test_macro_f1": round(test_m["macro_f1"], 4) if test_m else None,
        "test_top1": round(test_m["top1_acc"], 4) if test_m else None,
        "test_ece": round(test_m["ece"], 4) if test_m else None,
    }
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    overrides = {}
    config_fields = {f.name: f.type for f in dataclasses.fields(Config)}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ (cần định dạng key=val): {pair}")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in config_fields:
            raise ValueError(f"Config không có thuộc tính '{key}'. Các thuộc tính: {list(config_fields.keys())}")

        field_type = config_fields[key]

        if val.lower() in ("none", "null"):
            overrides[key] = None
        elif field_type in (bool, "bool"):
            overrides[key] = val.lower() in ("true", "1", "yes")
        elif field_type in (int, "int"):
            overrides[key] = int(val)
        elif field_type in (float, "float"):
            overrides[key] = float(val)
        else:
            overrides[key] = val

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0."""
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm huấn luyện Lab Day 2")
    parser.add_argument("--set", nargs="*", default=[], help="Cặp key=value ghi đè Config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    run(cfg)


if __name__ == "__main__":
    main()
