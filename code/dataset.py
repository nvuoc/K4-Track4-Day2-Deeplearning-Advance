"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)                      -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)              -> torchvision transform
    DeepWeedsDataset[i]                                 -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame (train_df, val_df, test_df).
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy đủ file chia dữ liệu cho fold {fold} tại {labels_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if not {"Filename", "Label"}.issubset(df.columns):
            raise ValueError(f"{name}_subset{fold}.csv thiếu cột bắt buộc: Filename hoặc Label")

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir` (nếu thư mục tồn tại)
    """
    images_dir = Path(images_dir)
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    # 1. Kiểm tra tổng số ảnh
    assert n_total == 17509, f"Tổng số ảnh phải đúng 17.509, nhận {n_total}"

    # 2. Kiểm tra giao giữa các tập phải rỗng
    tv = train_files & val_files
    tt = train_files & test_files
    vt = val_files & test_files
    assert len(tv) == 0, f"Giao train và val không rỗng: {len(tv)} ảnh trùng lặp"
    assert len(tt) == 0, f"Giao train và test không rỗng: {len(tt)} ảnh trùng lặp"
    assert len(vt) == 0, f"Giao val và test không rỗng: {len(vt)} ảnh trùng lặp"

    # 3. Hợp đủ 17509 ảnh
    union_files = train_files | val_files | test_files
    assert len(union_files) == 17509, f"Hợp ba tập phải có 17.509 ảnh duy nhất, nhận {len(union_files)}"

    # 4. Kiểm tra sự tồn tại của file ảnh
    missing_files = []
    if images_dir.exists():
        for fname in union_files:
            if not (images_dir / fname).exists():
                missing_files.append(fname)
                if len(missing_files) >= 5:
                    break
        if missing_files:
            raise FileNotFoundError(f"Có ảnh trong CSV không tồn tại trong {images_dir}: {missing_files}...")

    # Phân bố theo lớp
    per_class = {}
    for c in range(NUM_CLASSES):
        c_name = CLASS_NAMES[c]
        train_c = int((train_df["Label"] == c).sum())
        val_c = int((val_df["Label"] == c).sum())
        test_c = int((test_df["Label"] == c).sum())
        total_c = train_c + val_c + test_c
        per_class[c_name] = {
            "label": c,
            "train": train_c,
            "val": val_c,
            "test": test_c,
            "total": total_c,
        }

    stats = {
        "n": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": n_total,
            "ratios": (round(n_train / n_total, 4), round(n_val / n_total, 4), round(n_test / n_total, 4)),
        },
        "per_class": per_class,
        "overlap": {
            "train_val": len(tv),
            "train_test": len(tt),
            "val_test": len(vt),
        },
        "all_files_exist": len(missing_files) == 0,
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo torchvision transform.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: Resize 256 -> CenterCrop(img_size) (hoặc Resize(img_size) nếu img_size==256) + Normalize.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train:
        if img_size == 256:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ])

    # Chế độ huấn luyện (train = True)
    if aug == "basic":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug == "color":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug == "trivial":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug == "randaug":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Fallback to basic
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (image_tensor, label: int, filename: str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        label = self.labels[i]
        img_path = self.images_dir / fname

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            # Fallback tạo ảnh mẫu nếu file chưa tải (dùng cho testing/dry run)
            img = Image.new("RGB", (256, 256), color=(100, 150, 100))

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label), fname


def _worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader theo quy chuẩn của bài lab."""
    ds = DeepWeedsDataset(df=df, images_dir=images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            labels = df["Label"].to_numpy(dtype=np.int64)
            class_counts = np.bincount(labels, minlength=NUM_CLASSES)
            # Trọng số tỉ lệ nghịch với tần suất lớp
            class_weights_arr = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights_arr[labels]
            sampler_obj = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler_obj,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                worker_init_fn=_worker_init_fn,
            )
        else:
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                worker_init_fn=_worker_init_fn,
            )
    else:
        # Tập val / test: tuyệt đối không shuffle, giữ thứ tự chuẩn
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
