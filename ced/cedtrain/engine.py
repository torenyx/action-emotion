# -*- coding: utf-8 -*-
"""
训练/评估引擎：Focal Loss、单轮训练/评估、TTA 推理、K 折集成。
"""

from __future__ import annotations

import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TA
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .config import TrainConfig, TASK_META
from .modeling import CedAudioEmotionModel
from .data import AudioEmotionDataset

logger = logging.getLogger("cedtrain")

TASK_NAMES = list(TASK_META.keys())
TASK_NAME_TO_ID = {n: i for i, n in enumerate(TASK_NAMES)}
TASK_MASK_FNS = {
    n: (lambda tid, tw=TASK_NAME_TO_ID[n]: tid == tw)
    for n in TASK_NAMES
}


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) + 可选 Label Smoothing。"""

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal  # "none"：逐样本 (N,)


# ============================================================
# Warmup + CosineAnnealing 调度器
# ============================================================

def get_warmup_cosine_scheduler(
    optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01,
) -> LambdaLR:
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 特征空间 Mixup
# ============================================================

def batch_to_device(batch: dict, device: torch.device) -> dict:
    """仅将张量移到 device；Mixup 元数据（如 _mixup_lam 为 float）保持原样。"""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _focal_nll_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    """与 FocalLoss 一致，reduction=none，逐样本标量损失。"""
    ce = F.cross_entropy(
        logits, targets,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma) * ce


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    task_criteria: dict[str, FocalLoss],
    species_criterion: FocalLoss,
    task_loss_weights: dict[str, float],
    cfg: TrainConfig,
    is_train: bool,
    device: torch.device,
) -> torch.Tensor:
    """物种二分类 + 多任务情绪；训练期 Mixup 时对同一 task 内成对样本做软标签 Focal。"""
    task_ids = batch["task_id"].long()
    labels = batch["label"]
    mixup = is_train and cfg.mixup_alpha > 0 and "_mixup_perm" in batch
    perm = batch.get("_mixup_perm")
    same_task = batch.get("_mixup_same_task")
    lam = float(batch["_mixup_lam"]) if mixup else 1.0

    loss_species = species_criterion(outputs["species"], task_ids)
    loss = cfg.species_loss_weight * loss_species

    for task_name in TASK_NAMES:
        tid = TASK_NAME_TO_ID[task_name]
        mask = task_ids == tid
        if not mask.any():
            continue
        logits_m = outputs[task_name][mask]
        y_a = labels[mask]

        if mixup and perm is not None and same_task is not None:
            idx = torch.where(mask)[0]
            st = same_task[idx]
            y_b = labels[perm[idx]]
            focal_a = _focal_nll_per_sample(
                logits_m, y_a, cfg.focal_gamma, cfg.label_smoothing,
            )
            focal_b = _focal_nll_per_sample(
                logits_m, y_b, cfg.focal_gamma, cfg.label_smoothing,
            )
            lam_t = torch.tensor(lam, device=device, dtype=logits_m.dtype)
            mixed = lam_t * focal_a + (1.0 - lam_t) * focal_b
            t_loss = torch.where(st, mixed, focal_a).mean()
        else:
            t_loss = task_criteria[task_name](logits_m, y_a)

        w = task_loss_weights.get(task_name, 1.0)
        loss = loss + w * t_loss
    return loss


def _within_task_permutation(task_ids: torch.Tensor) -> torch.Tensor:
    """在 batch 内按 task_id 分组各自随机打乱，使配对始终同任务（避免狗/猫跨任务 Mixup 失效）。"""
    B = task_ids.shape[0]
    device = task_ids.device
    perm = torch.empty(B, dtype=torch.long, device=device)
    for tid in torch.unique(task_ids):
        idx = torch.where(task_ids == tid)[0]
        k = idx.numel()
        if k <= 1:
            perm[idx] = idx
        else:
            perm[idx] = idx[torch.randperm(k, device=device)]
    return perm


def mixup_mel(batch: dict, alpha: float) -> dict:
    """
    Mel 频谱空间 Mixup：只在同一 task_id 内做插值（perm 为组内随机置换）。
    对频谱做线性插值等价于对声学特征做 soft 数据增强。
    """
    if alpha <= 0:
        return batch

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    mel = batch["mel"]
    task_ids = batch["task_id"]
    perm = _within_task_permutation(task_ids)
    same_task = torch.ones(mel.shape[0], dtype=torch.bool, device=mel.device)

    mixed_mel = lam * mel + (1 - lam) * mel[perm]

    out = {k: v for k, v in batch.items()}
    out["mel"] = mixed_mel
    out["_mixup_perm"] = perm
    out["_mixup_lam"] = float(lam)
    out["_mixup_same_task"] = same_task
    return out


def spec_augment_mel(mel: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """SpecAugment：对 log-mel 做时间/频率掩码（与 AST 保持一致）。"""
    if not cfg.spec_augment_enabled:
        return mel
    B, _, _ = mel.shape
    out = mel.clone()
    for i in range(B):
        x = mel[i : i + 1]
        for _ in range(cfg.spec_aug_num_freq):
            x = TA.FrequencyMasking(cfg.spec_aug_freq_param)(x)
        for _ in range(cfg.spec_aug_num_time):
            x = TA.TimeMasking(cfg.spec_aug_time_param)(x)
        out[i] = x.squeeze(0)
    return out


class ModelEMA:
    """对可训练参数 + buffer（如 LayerNorm/BN 统计量）做指数滑动平均。"""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow_params = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
        }
        self.shadow_buffers = {
            name: b.detach().clone()
            for name, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            self.shadow_params[name].mul_(d).add_(p.data, alpha=1.0 - d)
        for name, b in model.named_buffers():
            if b.is_floating_point():
                self.shadow_buffers[name].mul_(d).add_(b.data, alpha=1.0 - d)
            else:
                self.shadow_buffers[name].copy_(b.data)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self._backup_params = {
            name: p.data.clone()
            for name, p in model.named_parameters()
        }
        self._backup_buffers = {
            name: b.data.clone()
            for name, b in model.named_buffers()
        }
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow_params[name])
        for name, b in model.named_buffers():
            b.data.copy_(self.shadow_buffers[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            p.data.copy_(self._backup_params[name])
        for name, b in model.named_buffers():
            b.data.copy_(self._backup_buffers[name])


def checkpoint_state_dict(
    model: CedAudioEmotionModel,
    ema: ModelEMA | None,
    cfg: TrainConfig,
) -> dict:
    """早停保存用：若启用 EMA，则导出 EMA 参数写入 state_dict（推理与旧版一致）。"""
    if cfg.use_ema and ema is not None:
        ema.apply_to(model)
        try:
            return {k: v.cpu().clone() for k, v in model.state_dict().items()}
        finally:
            ema.restore(model)
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# 单轮训练 / 评估
# ============================================================

def run_epoch(
    model: CedAudioEmotionModel,
    loader: DataLoader,
    task_criteria: dict[str, FocalLoss],
    species_criterion: FocalLoss,
    task_loss_weights: dict[str, float],
    cfg: TrainConfig,
    device: torch.device,
    optimizer=None,
    scaler: GradScaler | None = None,
    ema: ModelEMA | None = None,
) -> tuple[float, float, dict[str, float], dict[str, float], float]:
    """
    运行一轮训练或评估。

    Returns:
        (avg_loss, overall_acc, {task_name: task_acc}, {task_name: macro_f1}, species_acc)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    if not is_train and ema is not None:
        ema.apply_to(model)

    total_loss, num_batches = 0.0, 0
    task_correct = {t: 0 for t in TASK_NAMES}
    task_count = {t: 0 for t in TASK_NAMES}
    task_all_preds = {t: [] for t in TASK_NAMES}
    task_all_labels = {t: [] for t in TASK_NAMES}
    species_correct, species_total = 0, 0

    try:
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for batch in loader:
                batch = batch_to_device(batch, device)

                # 与 dinov3 少数侧更强增强一致：仅狗音频加轻微 Mel 噪声（task_id 由 TASK_NAME_TO_ID 决定）
                if is_train and cfg.dog_audio_mel_noise_std > 0:
                    tid = batch["task_id"]
                    mel = batch["mel"]
                    dog_mask = tid == TASK_NAME_TO_ID["dog_audio"]
                    if dog_mask.any():
                        mel = mel.clone()
                        mel[dog_mask] = mel[dog_mask] + torch.randn_like(mel[dog_mask]) * cfg.dog_audio_mel_noise_std
                        batch["mel"] = mel

                if is_train and cfg.mixup_alpha > 0:
                    batch = mixup_mel(batch, cfg.mixup_alpha)

                if is_train and cfg.spec_augment_enabled:
                    batch["mel"] = spec_augment_mel(batch["mel"], cfg)

                use_amp = cfg.use_amp and device.type == "cuda"
                with autocast("cuda", enabled=use_amp):
                    # 始终使用 GT 路由计算 Loss，确保 Loss 计算正确
                    outputs = model(batch["mel"], species=batch["task_id"])
                    loss = compute_multitask_loss(
                        outputs, batch, task_criteria, species_criterion,
                        task_loss_weights, cfg, is_train, device,
                    )
                    # 验证时，额外做一次端到端前向传播，用于计算真实的端到端评估指标（与测试对齐）
                    if not is_train:
                        outputs_eval = model(batch["mel"], species=None)
                    else:
                        outputs_eval = outputs

                if is_train:
                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                    if ema is not None:
                        ema.update(model)

                total_loss += loss.item()
                num_batches += 1

                with torch.no_grad():
                    task_ids = batch["task_id"]
                    labels = batch["label"]
                    pred_sp = outputs_eval["species"].argmax(dim=1)
                    species_correct += (pred_sp == task_ids).sum().item()
                    species_total += task_ids.numel()
                    for task_name in TASK_NAMES:
                        mask = TASK_MASK_FNS[task_name](task_ids)
                        if mask.any():
                            preds = outputs_eval[task_name][mask].argmax(dim=1)
                            correct = (preds == labels[mask]).sum().item()
                            task_correct[task_name] += correct
                            task_count[task_name] += mask.sum().item()
                            task_all_preds[task_name].append(preds.detach().cpu())
                            task_all_labels[task_name].append(labels[mask].detach().cpu())
    finally:
        if not is_train and ema is not None:
            ema.restore(model)

    avg_loss = total_loss / max(num_batches, 1)
    task_acc = {}
    for t in TASK_NAMES:
        task_acc[t] = task_correct[t] / max(task_count[t], 1)
    overall_correct = sum(task_correct.values())
    overall_count = sum(task_count.values())
    overall_acc = overall_correct / max(overall_count, 1)

    task_f1: dict[str, float] = {}
    for t in TASK_NAMES:
        if task_all_preds[t]:
            all_p = torch.cat(task_all_preds[t]).numpy()
            all_l = torch.cat(task_all_labels[t]).numpy()
            _, _, f1, _ = precision_recall_fscore_support(
                all_l, all_p, average="macro", zero_division=0,
            )
            task_f1[t] = float(f1)
        else:
            task_f1[t] = 0.0

    species_acc = species_correct / max(species_total, 1)
    return avg_loss, overall_acc, task_acc, task_f1, species_acc


# ============================================================
# 测试集完整评估
# ============================================================

@torch.no_grad()
def full_evaluation(
    model: CedAudioEmotionModel,
    loader: DataLoader,
    device: torch.device,
    dataset: AudioEmotionDataset,
    cfg: TrainConfig,
) -> dict:
    """返回 {task: (cm, class_names)} 混淆矩阵数据 + 打印评估报告。"""
    model.eval()

    all_preds = {t: [] for t in TASK_NAMES}
    all_targets = {t: [] for t in TASK_NAMES}
    all_sp_pred, all_sp_tgt = [], []

    for batch in loader:
        batch = batch_to_device(batch, device)
        use_amp = cfg.use_amp and device.type == "cuda"
        with autocast("cuda", enabled=use_amp):
            # 端到端：不泄露物种 GT，用预测物种做情绪头路由
            outputs = model(batch["mel"], species=None)

        task_ids = batch["task_id"]
        labels = batch["label"]
        pred_sp = outputs["species"].argmax(dim=1).cpu().numpy()
        all_sp_pred.append(pred_sp)
        all_sp_tgt.append(task_ids.cpu().numpy())

        for task_name in TASK_NAMES:
            mask = TASK_MASK_FNS[task_name](task_ids)
            if mask.any():
                preds = outputs[task_name][mask].argmax(dim=1).cpu().numpy()
                tgts = labels[mask].cpu().numpy()
                all_preds[task_name].append(preds)
                all_targets[task_name].append(tgts)

    report_lines = []
    cm_data = {}

    if all_sp_pred:
        sp_pred = np.concatenate(all_sp_pred)
        sp_tgt = np.concatenate(all_sp_tgt)
        sp_acc = (sp_pred == sp_tgt).mean() * 100
        report_lines.append(f"\n{'=' * 55}")
        report_lines.append(
            f"物种(狗/猫) 端到端准确率 (species=None 硬路由): {sp_acc:.2f}%",
        )
        report_lines.append(f"{'-' * 55}")

    for task_name in TASK_NAMES:
        if not all_preds[task_name]:
            continue
        preds = np.concatenate(all_preds[task_name])
        targets = np.concatenate(all_targets[task_name])
        class_names = dataset.task_class_names[task_name]

        p, r, f1, sup = precision_recall_fscore_support(
            targets, preds,
            average=None, labels=list(range(len(class_names))),
            zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0,
        )
        weighted_f1 = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0,
        )[2]
        acc = (preds == targets).mean() * 100

        report_lines.append(f"\n{'=' * 55}")
        report_lines.append(f"任务: {task_name}  (Accuracy={acc:.2f}%)")
        report_lines.append(f"{'-' * 55}")
        report_lines.append(
            f"{'类别':>14s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'支持数':>6s}"
        )
        for j, cls in enumerate(class_names):
            report_lines.append(
                f"  {cls:>12s}  {p[j]*100:>8.2f}%  {r[j]*100:>8.2f}%  "
                f"{f1[j]*100:>8.2f}%  {int(sup[j]):>6d}"
            )
        report_lines.append(f"{'-' * 55}")
        report_lines.append(
            f"  {'Macro Avg':>12s}  {macro_p*100:>8.2f}%  "
            f"{macro_r*100:>8.2f}%  {macro_f1*100:>8.2f}%"
        )
        report_lines.append(
            f"  {'Weighted F1':>12s}  {'':>9s}  {'':>9s}  {weighted_f1*100:>8.2f}%"
        )

        cm = confusion_matrix(
            targets, preds, labels=list(range(len(class_names))),
        )
        cm_data[task_name] = (cm, class_names)

    report = "\n".join(report_lines)
    logger.info(report)
    return cm_data


# ============================================================
# TTA 推理 + K 折集成
# ============================================================

def _tta_augment_mel(mel: torch.Tensor, step_idx: int, cfg: TrainConfig) -> torch.Tensor:
    """测试时增强：按步轮换 加性噪声 / 时间平移 / 频率掩码 / 时间掩码。"""
    B, n_mels, T = mel.shape
    device = mel.device
    pad = float(cfg.mel_pad_db)
    mode = step_idx % 4

    if mode == 0:
        return mel + torch.randn_like(mel) * cfg.tta_noise_std

    if mode == 1:
        out = mel.clone()
        if T <= 1:
            return out
        smax = min(cfg.tta_time_shift_max, max(0, T - 1))
        for i in range(B):
            s = int(torch.randint(-smax, smax + 1, (1,), device=device).item())
            out[i] = torch.roll(mel[i], shifts=s, dims=-1)
        return out

    if mode == 2:
        out = mel.clone()
        if n_mels <= 1:
            return out
        for i in range(B):
            f0 = int(torch.randint(0, n_mels, (1,), device=device).item())
            max_w = min(cfg.tta_freq_mask_max, n_mels - f0)
            if max_w < 1:
                continue
            w = int(torch.randint(1, max_w + 1, (1,), device=device).item())
            out[i, f0 : f0 + w, :] = pad
        return out

    out = mel.clone()
    if T <= 1:
        return out
    for i in range(B):
        t0 = int(torch.randint(0, T, (1,), device=device).item())
        max_w = min(cfg.tta_time_mask_max, T - t0)
        if max_w < 1:
            continue
        w = int(torch.randint(1, max_w + 1, (1,), device=device).item())
        out[i, :, t0 : t0 + w] = pad
    return out


@torch.no_grad()
def predict_tta(
    model: CedAudioEmotionModel,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    测试时增强：向 mel 频谱注入微弱高斯噪声 n_tta 次，对 softmax 概率取均值。
    """
    model.eval()
    task_probs = {t: [] for t in TASK_NAMES}
    task_targets = {t: [] for t in TASK_NAMES}

    for batch in loader:
        batch = batch_to_device(batch, device)
        task_ids = batch["task_id"]
        labels = batch["label"]

        task_masks = {}
        task_accum = {}
        for task_name in TASK_NAMES:
            mask = TASK_MASK_FNS[task_name](task_ids)
            if mask.any():
                n_cls = TASK_META[task_name]["num_classes"]
                task_masks[task_name] = mask
                task_accum[task_name] = torch.zeros(
                    mask.sum(), n_cls, device=device,
                )

        for step in range(cfg.tta_steps):
            noisy_mel = _tta_augment_mel(batch["mel"], step, cfg)
            use_amp = cfg.use_amp and device.type == "cuda"
            with autocast("cuda", enabled=use_amp):
                outputs = model(noisy_mel, species=None)
            for task_name, mask in task_masks.items():
                task_accum[task_name] += F.softmax(
                    outputs[task_name][mask].float(), dim=-1,
                )

        for task_name, mask in task_masks.items():
            task_probs[task_name].append(
                (task_accum[task_name] / cfg.tta_steps).cpu().numpy()
            )
            task_targets[task_name].append(labels[mask].cpu().numpy())

    return {
        t: (np.concatenate(task_probs[t]), np.concatenate(task_targets[t]))
        for t in TASK_NAMES if task_probs[t]
    }


@torch.no_grad()
def ensemble_evaluation(
    model_states: list[dict],
    loader: DataLoader,
    device: torch.device,
    dataset: AudioEmotionDataset,
    cfg: TrainConfig,
) -> dict:
    """K 折 × TTA 集成评估，返回混淆矩阵数据。"""
    model = CedAudioEmotionModel(cfg).to(device)
    fold_probs = {t: [] for t in TASK_NAMES}
    targets_ref = {t: None for t in TASK_NAMES}

    for fold_idx, state in enumerate(model_states):
        model.load_state_dict(state)
        fold_result = predict_tta(model, loader, device, cfg)
        logger.info(f"  折 {fold_idx + 1}/{len(model_states)} TTA 推理完成")
        for task_name, (probs, tgts) in fold_result.items():
            fold_probs[task_name].append(probs)
            if targets_ref[task_name] is None:
                targets_ref[task_name] = tgts

    report_lines = []
    cm_data = {}
    prob_bundle: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for task_name in TASK_NAMES:
        if not fold_probs[task_name]:
            continue
        avg_probs = np.mean(fold_probs[task_name], axis=0)
        preds = avg_probs.argmax(axis=1)
        targets = targets_ref[task_name]
        class_names = dataset.task_class_names[task_name]
        prob_bundle[task_name] = (avg_probs, targets)

        p, r, f1, sup = precision_recall_fscore_support(
            targets, preds,
            average=None, labels=list(range(len(class_names))),
            zero_division=0,
        )
        macro_f1 = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0,
        )[2]
        weighted_f1 = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0,
        )[2]
        acc = (preds == targets).mean() * 100

        report_lines.append(f"\n{'=' * 55}")
        report_lines.append(f"[集成] 任务: {task_name}  (Accuracy={acc:.2f}%)")
        report_lines.append(f"{'-' * 55}")
        report_lines.append(
            f"{'类别':>14s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'支持数':>6s}"
        )
        for j, cls in enumerate(class_names):
            report_lines.append(
                f"  {cls:>12s}  {p[j]*100:>8.2f}%  {r[j]*100:>8.2f}%  "
                f"{f1[j]*100:>8.2f}%  {int(sup[j]):>6d}"
            )
        report_lines.append(f"{'-' * 55}")
        report_lines.append(f"  Macro F1: {macro_f1*100:.2f}%  |  Weighted F1: {weighted_f1*100:.2f}%")

        cm = confusion_matrix(
            targets, preds, labels=list(range(len(class_names))),
        )
        cm_data[task_name] = (cm, class_names)

    logger.info("\n".join(report_lines))
    return cm_data, prob_bundle


# ============================================================
# 单折训练流程
# ============================================================

def train_fold(
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset: AudioEmotionDataset,
    train_idx: np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[dict, dict]:
    """
    训练单折，返回 (best_state_dict, history)。
    """
    from .utils import seed_everything
    seed_everything(cfg.seed + fold_idx)

    model = CedAudioEmotionModel(cfg).to(device)

    # 渐进解冻调度：按 epoch 升序排列，便于逐 epoch 查找当前策略
    unfreeze_schedule = sorted(cfg.unfreeze_schedule.items())  # [(ep, stage), ...]
    _current_stage: list[str] = [""]  # 用列表保持可变引用

    def _apply_unfreeze_if_needed(epoch: int) -> None:
        """若当前 epoch 触发新阶段，调用 set_unfreeze_stage 并重建 optimizer 参数组。"""
        stage = _current_stage[0]
        for ep_thresh, ep_stage in unfreeze_schedule:
            if epoch >= ep_thresh:
                stage = ep_stage
        if stage == _current_stage[0]:
            return
        _current_stage[0] = stage
        model.set_unfreeze_stage(stage)

        # 计算当前 scheduler 的衰减系数，确保新参数组以衰减后 LR 加入
        last_lrs = scheduler.get_last_lr()
        head_base = scheduler.base_lrs[-1] if scheduler.base_lrs else cfg.head_lr
        lr_scale = last_lrs[-1] / max(head_base, 1e-12) if epoch > 1 else 1.0

        new_groups = model.get_param_groups(cfg)
        for g in new_groups:
            g["initial_lr"] = g["lr"]
            g["lr"] = g["lr"] * lr_scale

        old_state = {id(p): optimizer.state[p] for group in optimizer.param_groups for p in group['params']}
        optimizer.param_groups.clear()
        optimizer.state.clear()
        for g in new_groups:
            optimizer.add_param_group(g)
            for p in g['params']:
                if id(p) in old_state:
                    optimizer.state[p] = old_state[id(p)]

        scheduler.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]

    param_groups = model.get_param_groups(cfg)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_warmup_cosine_scheduler(
        optimizer, cfg.warmup_epochs, cfg.num_epochs, cfg.min_lr_ratio,
    )
    scaler = (
        GradScaler("cuda")
        if (cfg.use_amp and device.type == "cuda")
        else None
    )

    ema = ModelEMA(model, cfg.ema_decay) if cfg.use_ema else None

    # WeightedRandomSampler 已做逆频均衡；Focal 不再传 class weight；两任务等权（对齐 dinov3）
    species_criterion = FocalLoss(
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    )
    task_criteria = {
        t: FocalLoss(
            gamma=cfg.focal_gamma,
            label_smoothing=cfg.label_smoothing,
        )
        for t in TASK_NAMES
    }
    task_loss_weights = {t: 1.0 for t in TASK_NAMES}
    logger.info(
        f"  [折{fold_idx+1}] 损失: 物种 Focal ×{cfg.species_loss_weight} + 情绪任务等权 1.0；"
        f"狗音频 Mel 噪声 std={cfg.dog_audio_mel_noise_std}；"
        f"SpecAugment={'开' if cfg.spec_augment_enabled else '关'}；"
        f"EMA={'decay=' + str(cfg.ema_decay) if ema else '关'}；"
        f"渐进解冻调度={dict(unfreeze_schedule)}",
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_balanced_macro_f1": [], "val_balanced_macro_f1": [], "lr": [],
    }
    for t in TASK_NAMES:
        history[f"train_macro_f1_{t}"] = []
        history[f"val_macro_f1_{t}"] = []

    best_metric = 0.0
    patience_count = 0
    best_state = None
    t_start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        t_ep = time.time()

        # 渐进解冻：epoch 开始前检查是否需要切换解冻阶段
        _apply_unfreeze_if_needed(epoch)

        current_lr = optimizer.param_groups[-1]["lr"]  # 取 head 参数组 LR 作为展示值

        tr_loss, tr_acc, tr_task_acc, tr_task_f1, tr_sp_acc = run_epoch(
            model, train_loader, task_criteria, species_criterion, task_loss_weights,
            cfg, device, optimizer, scaler, ema,
        )
        val_loss, val_acc, val_task_acc, val_task_f1, val_sp_acc = run_epoch(
            model, val_loader, task_criteria, species_criterion, task_loss_weights,
            cfg, device, None, None, ema,
        )
        scheduler.step()

        train_balanced = float(
            np.mean([tr_task_f1[t] for t in TASK_NAMES]),
        )
        val_balanced = float(
            np.mean([val_task_f1[t] for t in TASK_NAMES]),
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_balanced_macro_f1"].append(train_balanced)
        history["val_balanced_macro_f1"].append(val_balanced)
        history["lr"].append(current_lr)
        for t in TASK_NAMES:
            history[f"train_macro_f1_{t}"].append(tr_task_f1.get(t, 0.0))
            history[f"val_macro_f1_{t}"].append(val_task_f1.get(t, 0.0))

        elapsed = time.time() - t_ep
        task_str = " | ".join(
            f"{t}:F1宏={val_task_f1.get(t, 0)*100:.1f}%"
            for t in TASK_NAMES
        )
        logger.info(
            f"[折{fold_idx+1}] Epoch [{epoch:>3d}/{cfg.num_epochs}] "
            f"训练: loss={tr_loss:.4f} | "
            f"验证: loss={val_loss:.4f} balanced_macro_F1={val_balanced*100:.2f}% | "
            f"物种acc={val_sp_acc*100:.1f}% (GT路由) | "
            f"{task_str} | lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_balanced > best_metric + 1e-4:
            best_metric = val_balanced
            patience_count = 0
            best_state = checkpoint_state_dict(model, ema, cfg)
            logger.info(
                f"  >>> 最优权重已更新 (balanced_macro_F1={val_balanced*100:.2f}%，"
                f"{'EMA' if ema else '当前模型'}参数)",
            )
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                logger.info(
                    f"  早停触发: 验证 balanced_macro_F1 连续 {cfg.patience} 轮未改善",
                )
                break

    elapsed_total = time.time() - t_start
    logger.info(
        f"[折{fold_idx+1}] 训练完成: {len(history['train_loss'])} 轮, "
        f"最优 balanced_macro_F1={best_metric*100:.2f}%, 用时 {elapsed_total/60:.1f} 分钟",
    )
    return best_state, history
