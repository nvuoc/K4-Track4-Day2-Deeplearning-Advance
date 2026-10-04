"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                      -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)     -> list[dict] cho optimizer
    count_params(model)                                         -> float (triệu)
    count_gmacs(model, img_size)                                -> float (GMAC)
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "mobilenetv3": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = pretrained if init != "scratch" else False

    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Lưu lại thông tin pretrained tag
    pretrained_cfg = getattr(model, "pretrained_cfg", {})
    tag = pretrained_cfg.get("tag", "scratch" if not is_pretrained else "default")
    model.pretrained_tag = f"{name}.{tag}"
    model.backbone_name = name
    model.init_mode = init

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.

    Lưu ý: khi đóng băng backbone, BatchNorm cũng phải ở chế độ eval.
    """
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters())

    for param in model.parameters():
        if param in classifier_params:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.backbone_frozen = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    1. Backbone trọng số nhiều chiều (ndim > 1): lr = lr_backbone, weight_decay = weight_decay
    2. Backbone norm và bias (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    3. Head mới: lr = lr_head (gấp 10 lần backbone), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    head_params = set(classifier.parameters())

    bb_decay = []
    bb_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = param in head_params
        # Norm và bias thường có ndim <= 1
        is_no_decay = (param.ndim <= 1) or ("bias" in name) or ("norm" in name.lower()) or ("bn" in name.lower())

        if is_head:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if is_no_decay:
                bb_no_decay.append(param)
            else:
                bb_decay.append(param)

    groups = []
    if bb_decay:
        groups.append({"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if bb_no_decay:
        groups.append({"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Tính GMAC cho một ảnh 3 x img_size x img_size bằng forward hooks."""
    total_macs = [0]
    hooks = []

    def conv_hook(module, inp, out):
        # input: (N, Cin, Hin, Win), output: (N, Cout, Hout, Wout)
        out_h, out_w = out.shape[2], out.shape[3]
        cin = module.in_channels
        cout = module.out_channels
        kh, kw = module.kernel_size
        groups = module.groups
        macs = (cin * cout * kh * kw * out_h * out_w) // groups
        total_macs[0] += macs

    def linear_hook(module, inp, out):
        in_features = module.in_features
        out_features = module.out_features
        total_macs[0] += in_features * out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    dummy_input = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    gmacs = total_macs[0] / 1e9
    return round(gmacs, 3)
