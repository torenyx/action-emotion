# -*- coding: utf-8 -*-
"""
大创项目：基于 LanguageBind 统一隐空间与多任务路由网络的多模态猫狗情绪识别
正式训练脚本（学术级）

与 ImageBind 版（1.py）的关键差异：
  - 输入特征维度：768（LanguageBind projection_dim）vs 1024（ImageBind）
  - 特征来源：LanguageBind Image + Audio-FT（ICLR 2024）
  - 隐藏层维度适配调整：384（匹配 768 的输入，保持 2:1 压缩比）
  - 数据处理（与 dinov3 / train/1.py 对齐）：仅 WeightedRandomSampler + Focal(无 class weight) + 任务等权；
    避免采样器与逆频 loss 双重补偿；物种辅助损失 0.3×
  - 数据分布：图像狗多猫少、音频猫多狗少 → 特征噪声仅对「少数侧」加强（猫图 / 狗音频）
  - 早停与最优权重：balanced_macro_F1（四任务 macro-F1 平均）

运行：
  conda activate d2l
  python train/1_languagebind.py

输入：  data/features_languagebind_npy/*.npy（由 extract_features_languagebind.py 预先提取）
输出：
  moxing/LB_MultiBranchMLP_{时间戳}.pkl
  figure/lb_train_curve_{时间戳}.png
  figure/lb_confusion_split_image_{时间戳}.png、figure/lb_confusion_split_audio_{时间戳}.png
  figure/lb_interpret_*_{时间戳}.png（各类别 P/R/F1、ROC、置信度、Top 混淆）
  txt/lb_train_log_{时间戳}.txt
"""

import os
import sys
import time
import logging
import platform
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# 路径配置
# ============================================================
ROOT      = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FEAT_DIR  = ROOT / "data" / "features_languagebind_npy"
MODEL_DIR = ROOT / "moxing"
FIG_DIR   = ROOT / "figure"
TXT_DIR   = ROOT / "txt"
for d in [MODEL_DIR, FIG_DIR, TXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d%H%M%S")
log_path = TXT_DIR / f"lb_train_log_{TS}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# 超参数
# ============================================================
SEED          = 42
BATCH_SIZE    = 256
NUM_EPOCHS    = 120
LR            = 5e-4
WEIGHT_DECAY  = 2e-4
INPUT_DIM     = 768             # LanguageBind projection_dim
HIDDEN_DIM    = 384             # 保持 2:1 压缩比（768→384）
PATIENCE      = 20
WARMUP_EPOCHS = 5
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1
MIXUP_ALPHA   = 0.3
FOCAL_GAMMA   = 2.0
LABEL_SMOOTH  = 0.05
NOISE_STD     = 0.05
# 少数模态侧隐层噪声倍数：图像强化猫、音频强化狗（与 dinov3 猫强增强 / 少数类补多样性同思路）
MINORITY_FEATURE_NOISE_MULT = 1.5

N_FOLDS       = 1            # >1 则启用 K 折集成，1 = 单次训练（已关闭交叉验证）
TTA_STEPS     = 10
TTA_NOISE_STD = 0.03

TASK_DROPOUTS = {
    "dog_img":   0.25,   # 图像狗样本多
    "cat_img":   0.40,   # 图像猫样本少 → 更强正则
    "dog_audio": 0.32,   # 音频狗少（约 1200）→ 提高 Dropout 防过拟合
    "cat_audio": 0.26,   # 音频猫多（约 2900）
}

# ============================================================
# 可复现性
# ============================================================
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ============================================================
# 标签映射
# ============================================================
DOG_IMG_CLASSES   = ["angry", "happy", "relaxed", "sad"]
CAT_IMG_CLASSES   = ["Angry", "Disgusted", "Happy", "Normal", "Sad", "Scared", "Surprised"]
DOG_AUDIO_CLASSES = ["barking", "growling", "howling", "whining"]
CAT_AUDIO_CLASSES = ["Angry", "Defence", "Fighting", "Happy", "HuntingMind",
                     "Mating", "MotherCall", "Paining", "Resting", "Warning"]

TASK_META = {
    "dog_img":   {"species": 0, "modality": 0, "classes": DOG_IMG_CLASSES,   "prefix": "dog_img"},
    "cat_img":   {"species": 1, "modality": 0, "classes": CAT_IMG_CLASSES,   "prefix": "cat_img"},
    "dog_audio": {"species": 0, "modality": 1, "classes": DOG_AUDIO_CLASSES, "prefix": "dog_audio"},
    "cat_audio": {"species": 1, "modality": 1, "classes": CAT_AUDIO_CLASSES, "prefix": "cat_audio"},
}

# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0,
                 weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ============================================================
# 数据集
# ============================================================
class MultimodalFeatureDataset(Dataset):
    """读取 LanguageBind 离线 .npy 特征，整合四个数据源并构建联合标签。"""

    def __init__(self, feat_dir: Path):
        all_feats, all_labels, all_species, all_modality, all_task_ids = [], [], [], [], []
        self.task_class_names = {}

        for task_id, (task_name, meta) in enumerate(TASK_META.items()):
            feat_path  = feat_dir / f"{meta['prefix']}_feat.npy"
            label_path = feat_dir / f"{meta['prefix']}_label.npy"

            if not feat_path.exists() or not label_path.exists():
                logger.error(f"特征文件缺失：{feat_path} 或 {label_path}")
                sys.exit(1)

            feats  = np.load(feat_path).astype(np.float32)
            labels = np.load(label_path).astype(np.int64)
            N = len(feats)
            n_cls = len(meta["classes"])

            assert feats.shape[1] == INPUT_DIM, (
                f"{task_name} 特征维度不匹配：期望 {INPUT_DIM}，实际 {feats.shape[1]}。"
                f"请确认使用的是 LanguageBind 提取的特征。"
            )
            assert labels.min() >= 0 and labels.max() < n_cls, (
                f"{task_name} 标签越界：min={labels.min()}, max={labels.max()}, n_cls={n_cls}"
            )

            all_feats.append(feats)
            all_labels.append(labels)
            all_species.append(np.full(N, meta["species"], dtype=np.int64))
            all_modality.append(np.full(N, meta["modality"], dtype=np.int64))
            all_task_ids.append(np.full(N, task_id, dtype=np.int64))
            self.task_class_names[task_name] = meta["classes"]

            logger.info(f"  加载 {task_name}: {N} 条，{n_cls} 类，维度 {feats.shape[1]}")

        self.features = torch.from_numpy(np.concatenate(all_feats))
        self.labels   = torch.from_numpy(np.concatenate(all_labels))
        self.species  = torch.from_numpy(np.concatenate(all_species))
        self.modality = torch.from_numpy(np.concatenate(all_modality))
        self.task_ids = torch.from_numpy(np.concatenate(all_task_ids))

        self.stratify_labels = (self.task_ids * 100 + self.labels).numpy()

        logger.info(f"  数据集合并完毕：共 {len(self.features)} 条样本，特征维度 {INPUT_DIM}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "feature":  self.features[idx],
            "label":    self.labels[idx],
            "species":  self.species[idx],
            "modality": self.modality[idx],
            "task_id":  self.task_ids[idx],
        }


# ============================================================
# 分层采样划分 + WeightedRandomSampler
# ============================================================
def stratified_split(dataset: MultimodalFeatureDataset, train_ratio, val_ratio, seed):
    labels = dataset.stratify_labels
    n = len(labels)

    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), labels))

    trainval_labels = labels[trainval_idx]
    relative_val = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_sub, val_sub = next(sss2.split(np.zeros(len(trainval_idx)), trainval_labels))

    train_idx = trainval_idx[train_sub]
    val_idx   = trainval_idx[val_sub]

    return train_idx, val_idx, test_idx


def make_weighted_sampler(dataset: MultimodalFeatureDataset, indices: np.ndarray) -> WeightedRandomSampler:
    strat = dataset.stratify_labels[indices]
    counter = Counter(strat)
    weights = np.array([1.0 / counter[s] for s in strat], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)


# ============================================================
# Mixup（特征空间）
# ============================================================
def mixup_batch(batch: dict, alpha: float) -> dict:
    if alpha <= 0:
        return batch

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    features = batch["feature"]
    labels   = batch["label"]
    task_ids = batch["task_id"]
    B = features.shape[0]

    perm = torch.randperm(B, device=features.device)
    same_task = (task_ids == task_ids[perm])

    mixed_features = features.clone()
    mixed_features[same_task] = lam * features[same_task] + (1 - lam) * features[perm][same_task]

    batch_out = {k: v for k, v in batch.items()}
    batch_out["feature"] = mixed_features
    batch_out["_mixup_perm"] = perm
    batch_out["_mixup_lam"]  = lam
    batch_out["_mixup_same_task"] = same_task
    return batch_out


# ============================================================
# 模型：残差多分支路由 MLP（适配 768 维输入）
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(x + self.net(x)))


class TaskHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        mid = in_dim // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiBranchRoutingMLP(nn.Module):
    """
    共享残差骨干 + 多层任务专属分类头。

    结构：
      Input(768) → L2Norm → 降维(hidden=384) → GaussNoise(train, 少数侧更强) → ResBlock × 2 → TaskHead × 4
    少数侧：图像 modality=0 且猫 species=1；音频 modality=1 且狗 species=0。
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.noise_std = NOISE_STD
        self.minority_noise_mult = MINORITY_FEATURE_NOISE_MULT
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=0.2),
            ResidualBlock(hidden_dim, dropout=0.2),
        )
        self.head_species   = nn.Linear(hidden_dim, 2)
        self.head_dog_img   = TaskHead(hidden_dim, len(DOG_IMG_CLASSES),   TASK_DROPOUTS["dog_img"])
        self.head_cat_img   = TaskHead(hidden_dim, len(CAT_IMG_CLASSES),   TASK_DROPOUTS["cat_img"])
        self.head_dog_audio = TaskHead(hidden_dim, len(DOG_AUDIO_CLASSES), TASK_DROPOUTS["dog_audio"])
        self.head_cat_audio = TaskHead(hidden_dim, len(CAT_AUDIO_CLASSES), TASK_DROPOUTS["cat_audio"])

    def forward(
        self,
        x: torch.Tensor,
        species: Optional[torch.Tensor] = None,
        modality: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.stem(F.normalize(x, p=2, dim=-1))
        if self.training and self.noise_std > 0:
            if species is not None and modality is not None:
                s = species.view(-1, 1)
                m = modality.view(-1, 1)
                # 图像狗多猫少 → 猫图更强扰动；音频猫多狗少 → 狗音频更强扰动
                minority = ((m == 0) & (s == 1)) | ((m == 1) & (s == 0))
                scale = torch.where(
                    minority,
                    torch.full_like(h, self.noise_std * self.minority_noise_mult),
                    torch.full_like(h, self.noise_std),
                )
                h = h + torch.randn_like(h) * scale
            else:
                h = h + torch.randn_like(h) * self.noise_std
        h = self.res_blocks(h)
        out: dict[str, torch.Tensor] = {
            "species":   self.head_species(h),
            "dog_img":   self.head_dog_img(h),
            "cat_img":   self.head_cat_img(h),
            "dog_audio": self.head_dog_audio(h),
            "cat_audio": self.head_cat_audio(h),
        }
        if return_hidden:
            out["hidden"] = h
        return out


# ============================================================
# Warmup + CosineAnnealing 学习率调度
# ============================================================
def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 任务路由（与 dinov3：采样器均衡后不再在 loss 里叠逆频权重）
# ============================================================
TASK_HEADS = ["dog_img", "cat_img", "dog_audio", "cat_audio"]
TASK_MASKS = {
    "dog_img":   lambda s, m: (s == 0) & (m == 0),
    "cat_img":   lambda s, m: (s == 1) & (m == 0),
    "dog_audio": lambda s, m: (s == 0) & (m == 1),
    "cat_audio": lambda s, m: (s == 1) & (m == 1),
}


# ============================================================
# 损失计算
# ============================================================
def compute_loss(
    outputs: dict, batch: dict,
    species_criterion: nn.Module,
    task_criteria: dict[str, FocalLoss],
    task_loss_weights: dict = None,
    mixup_active: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    species  = batch["species"]
    modality = batch["modality"]
    labels   = batch["label"]

    loss_species = species_criterion(outputs["species"], species)
    loss_total   = 0.3 * loss_species
    loss_dict    = {"species": loss_species.item()}

    lam       = batch.get("_mixup_lam", 1.0)
    perm      = batch.get("_mixup_perm", None)
    same_task = batch.get("_mixup_same_task", None)

    for task in TASK_HEADS:
        mask = TASK_MASKS[task](species, modality)
        if not mask.any():
            continue
        logits_t = outputs[task][mask]
        labels_t = labels[mask]

        if mixup_active and perm is not None and same_task is not None:
            task_same = same_task[mask]
            perm_labels_global = labels[perm][mask]
            n_cls = logits_t.shape[1]

            safe_perm_labels = perm_labels_global.clone()
            safe_perm_labels[~task_same] = 0
            safe_perm_labels = safe_perm_labels.clamp(0, n_cls - 1)

            ce_a = F.cross_entropy(logits_t, labels_t,
                                   weight=task_criteria[task].weight,
                                   reduction="none",
                                   label_smoothing=task_criteria[task].label_smoothing)
            pt_a = torch.exp(-ce_a)
            focal_a = ((1 - pt_a) ** task_criteria[task].gamma) * ce_a

            ce_b = F.cross_entropy(logits_t, safe_perm_labels,
                                   weight=task_criteria[task].weight,
                                   reduction="none",
                                   label_smoothing=task_criteria[task].label_smoothing)
            pt_b = torch.exp(-ce_b)
            focal_b = ((1 - pt_b) ** task_criteria[task].gamma) * ce_b

            per_sample = torch.where(task_same,
                                     lam * focal_a + (1 - lam) * focal_b,
                                     focal_a)
            t_loss = per_sample.mean()
        else:
            t_loss = task_criteria[task](logits_t, labels_t)

        w = task_loss_weights.get(task, 1.0) if task_loss_weights else 1.0
        loss_total = loss_total + w * t_loss
        loss_dict[task] = t_loss.item()

    return loss_total, loss_dict


# ============================================================
# 准确率计算
# ============================================================
def compute_accuracy_detailed(outputs: dict, batch: dict) -> dict[str, tuple[int, int]]:
    species  = batch["species"]
    modality = batch["modality"]
    labels   = batch["label"]

    result = {}
    for task, mask_fn in TASK_MASKS.items():
        mask = mask_fn(species, modality)
        if mask.any():
            preds = outputs[task][mask].argmax(dim=1)
            correct = (preds == labels[mask]).sum().item()
            count   = mask.sum().item()
            result[task] = (correct, count)
    return result


# ============================================================
# 训练 / 评估一轮
# ============================================================
def run_epoch(
    model, loader, species_criterion, task_criteria, optimizer=None, device="cpu",
    mixup_alpha: float = 0.0, task_loss_weights: dict = None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_count = 0.0, 0
    task_correct = {t: 0 for t in TASK_HEADS}
    task_count   = {t: 0 for t in TASK_HEADS}
    task_all_preds  = {t: [] for t in TASK_HEADS}
    task_all_labels = {t: [] for t in TASK_HEADS}

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            B = len(batch["feature"])

            if is_train and mixup_alpha > 0:
                batch = mixup_batch(batch, mixup_alpha)

            outputs = model(batch["feature"], batch["species"], batch["modality"])
            loss, ld = compute_loss(
                outputs, batch, species_criterion, task_criteria,
                task_loss_weights=task_loss_weights,
                mixup_active=(is_train and mixup_alpha > 0),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss  += loss.item() * B
            total_count += B

            species  = batch["species"]
            modality = batch["modality"]
            labels   = batch["label"]
            for t in TASK_HEADS:
                mask = TASK_MASKS[t](species, modality)
                if mask.any():
                    preds = outputs[t][mask].argmax(dim=1)
                    task_all_preds[t].append(preds.detach().cpu())
                    task_all_labels[t].append(labels[mask].detach().cpu())

            acc_d = compute_accuracy_detailed(outputs, batch)
            for t, (c, n) in acc_d.items():
                task_correct[t] += c
                task_count[t]   += n

    avg_loss = total_loss / total_count
    task_acc = {}
    for t in TASK_HEADS:
        task_acc[t] = task_correct[t] / task_count[t] if task_count[t] > 0 else 0.0
    overall_correct = sum(task_correct.values())
    overall_count   = sum(task_count.values())
    overall_acc = overall_correct / overall_count if overall_count > 0 else 0.0

    task_f1 = {}
    for t in TASK_HEADS:
        if task_all_preds[t]:
            all_p = torch.cat(task_all_preds[t]).numpy()
            all_l = torch.cat(task_all_labels[t]).numpy()
            _, _, f1, _ = precision_recall_fscore_support(
                all_l, all_p, average="macro", zero_division=0,
            )
            task_f1[t] = float(f1)
        else:
            task_f1[t] = 0.0

    return avg_loss, overall_acc, task_acc, task_f1


# ============================================================
# 测试集完整评估
# ============================================================
@torch.no_grad()
def full_evaluation(model, loader, device, dataset: MultimodalFeatureDataset):
    model.eval()

    all_preds   = {t: [] for t in TASK_HEADS}
    all_targets = {t: [] for t in TASK_HEADS}

    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["feature"], batch["species"], batch["modality"])
        species  = batch["species"]
        modality = batch["modality"]
        labels   = batch["label"]

        for task, mask_fn in TASK_MASKS.items():
            mask = mask_fn(species, modality)
            if mask.any():
                preds = outputs[task][mask].argmax(dim=1).cpu().numpy()
                tgts  = labels[mask].cpu().numpy()
                all_preds[task].append(preds)
                all_targets[task].append(tgts)

    report_lines = []
    cm_data = {}

    for task in TASK_HEADS:
        if not all_preds[task]:
            continue
        preds   = np.concatenate(all_preds[task])
        targets = np.concatenate(all_targets[task])
        class_names = dataset.task_class_names[task]

        p, r, f1, sup = precision_recall_fscore_support(
            targets, preds, average=None, labels=list(range(len(class_names))), zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0
        )
        weighted_f1 = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0
        )[2]
        acc = (preds == targets).mean() * 100

        report_lines.append(f"\n{'═' * 50}")
        report_lines.append(f"任务: {task}  (Accuracy={acc:.2f}%)")
        report_lines.append(f"{'─' * 50}")
        report_lines.append(f"{'类别':>14s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'支持数':>6s}")

        for j, cls in enumerate(class_names):
            report_lines.append(
                f"  {cls:>12s}  {p[j]*100:>8.2f}%  {r[j]*100:>8.2f}%  {f1[j]*100:>8.2f}%  {int(sup[j]):>6d}"
            )
        report_lines.append(f"{'─' * 50}")
        report_lines.append(f"  {'Macro Avg':>12s}  {macro_p*100:>8.2f}%  {macro_r*100:>8.2f}%  {macro_f1*100:>8.2f}%")
        report_lines.append(f"  {'Weighted F1':>12s}  {'':>9s}  {'':>9s}  {weighted_f1*100:>8.2f}%")

        cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
        cm_data[task] = (cm, class_names)

    report = "\n".join(report_lines)
    logger.info(report)
    return cm_data


# ============================================================
# TTA 推理 + K 折集成评估
# ============================================================
@torch.no_grad()
def predict_one_model_tta(
    model: nn.Module, loader: DataLoader, device,
    n_tta: int, tta_noise: float,
) -> dict:
    model.eval()
    task_probs_buf  = {t: [] for t in TASK_HEADS}
    task_target_buf = {t: [] for t in TASK_HEADS}

    for batch in loader:
        batch    = {k: v.to(device) for k, v in batch.items()}
        species  = batch["species"]
        modality = batch["modality"]
        labels   = batch["label"]

        task_masks = {}
        task_accum = {}
        for task, mask_fn in TASK_MASKS.items():
            mask = mask_fn(species, modality)
            if mask.any():
                n_cls = len(TASK_META[task]["classes"])
                task_masks[task] = mask
                task_accum[task] = torch.zeros(mask.sum(), n_cls, device=device)

        for _ in range(n_tta):
            noisy   = batch["feature"] + torch.randn_like(batch["feature"]) * tta_noise
            outputs = model(noisy, species, modality)
            for task, mask in task_masks.items():
                task_accum[task] += F.softmax(outputs[task][mask], dim=-1)

        for task, mask in task_masks.items():
            task_probs_buf[task].append((task_accum[task] / n_tta).cpu().numpy())
            task_target_buf[task].append(labels[mask].cpu().numpy())

    return {
        task: (np.concatenate(task_probs_buf[task]),
               np.concatenate(task_target_buf[task]))
        for task in TASK_HEADS if task_probs_buf[task]
    }


@torch.no_grad()
def full_evaluation_ensemble(
    model_states: list, loader: DataLoader, device,
    dataset: MultimodalFeatureDataset,
    n_tta: int = 10, tta_noise: float = 0.03,
) -> tuple[dict, dict]:
    model        = MultiBranchRoutingMLP().to(device)
    fold_probs   = {t: [] for t in TASK_HEADS}
    targets_ref  = {t: None for t in TASK_HEADS}

    for fold_idx, state in enumerate(model_states):
        model.load_state_dict(state)
        fold_result = predict_one_model_tta(model, loader, device, n_tta, tta_noise)
        logger.info(f"  折 {fold_idx + 1}/{len(model_states)} TTA 推理完成")
        for task, (probs, tgts) in fold_result.items():
            fold_probs[task].append(probs)
            if targets_ref[task] is None:
                targets_ref[task] = tgts

    report_lines = []
    cm_data      = {}

    for task in TASK_HEADS:
        if not fold_probs[task]:
            continue
        avg_probs   = np.mean(fold_probs[task], axis=0)
        preds       = avg_probs.argmax(axis=1)
        targets     = targets_ref[task]
        class_names = dataset.task_class_names[task]

        p, r, f1, sup = precision_recall_fscore_support(
            targets, preds, average=None,
            labels=list(range(len(class_names))), zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0,
        )
        weighted_f1 = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0,
        )[2]
        acc = (preds == targets).mean() * 100

        report_lines.append(f"\n{'═' * 50}")
        report_lines.append(f"任务: {task}  (Accuracy={acc:.2f}%)")
        report_lines.append(f"{'─' * 50}")
        report_lines.append(f"{'类别':>14s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'支持数':>6s}")
        for j, cls in enumerate(class_names):
            report_lines.append(
                f"  {cls:>12s}  {p[j]*100:>8.2f}%  {r[j]*100:>8.2f}%  {f1[j]*100:>8.2f}%  {int(sup[j]):>6d}"
            )
        report_lines.append(f"{'─' * 50}")
        report_lines.append(f"  {'Macro Avg':>12s}  {macro_p*100:>8.2f}%  {macro_r*100:>8.2f}%  {macro_f1*100:>8.2f}%")
        report_lines.append(f"  {'Weighted F1':>12s}  {'':>9s}  {'':>9s}  {weighted_f1*100:>8.2f}%")

        cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
        cm_data[task] = (cm, class_names)

    prob_bundle = {}
    for task in TASK_HEADS:
        if not fold_probs[task]:
            continue
        avg_probs = np.mean(fold_probs[task], axis=0)
        tg = targets_ref[task]
        if tg is not None:
            prob_bundle[task] = (avg_probs, tg)

    logger.info("\n".join(report_lines))
    return cm_data, prob_bundle


# ============================================================
# 可视化：训练曲线
# ============================================================
def _get_zh_font():
    zh_fonts = [f.fname for f in fm.fontManager.ttflist
                if any(kw in f.name for kw in
                       ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong"])]
    if zh_fonts:
        return fm.FontProperties(fname=zh_fonts[0])
    return fm.FontProperties()


_TRAIN_LEGEND_FS = 22
_TRAIN_TICK_FS = 20
_TRAIN_AXIS_FS = 20
_TASK_LEGEND_LOC = {
    "dog_img": "upper left", "cat_img": "lower right",
    "dog_audio": "upper left", "cat_audio": "lower right",
}
def plot_training_curves(history: dict, save_path: Path):
    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(28, 13))
    fig.patch.set_facecolor("white")
    line_kw = dict(linewidth=2.2, marker="o", markersize=3)

    colors = {"train": "#2E86AB", "val": "#E84855"}
    task_zh = {"dog_img": "狗图像", "cat_img": "猫图像",
               "dog_audio": "狗音频", "cat_audio": "猫音频"}

    def style_ax(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.2)

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color=colors["train"], label="训练", **line_kw)
    ax.plot(epochs, history["val_loss"],   color=colors["val"],   label="验证", **line_kw)
    ax.set_xlabel("Epoch", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.set_ylabel("Loss",  fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.legend(fontsize=_TRAIN_LEGEND_FS, prop=fp, framealpha=0.92)
    ax.tick_params(labelsize=_TRAIN_TICK_FS)
    style_ax(ax)

    tr_bf = history.get("train_balanced_macro_f1", history.get("train_acc", []))
    vl_bf = history.get("val_balanced_macro_f1", history.get("val_acc", []))
    ax = axes[0, 1]
    ax.plot(epochs, [a * 100 for a in tr_bf], color=colors["train"], label="训练", **line_kw)
    ax.plot(epochs, [a * 100 for a in vl_bf], color=colors["val"], label="验证", **line_kw)
    ax.set_xlabel("Epoch", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.set_ylabel("平衡宏 F1 (%)", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.legend(fontsize=_TRAIN_LEGEND_FS, prop=fp, framealpha=0.92)
    ax.tick_params(labelsize=_TRAIN_TICK_FS)
    style_ax(ax)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, max(y1 * 1.08, y1 + 5))

    ax = axes[0, 2]
    ax.plot(epochs, history["lr"], color="#6C5B7B", linewidth=2.5)
    ax.set_xlabel("Epoch", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.set_ylabel("学习率", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.tick_params(labelsize=_TRAIN_TICK_FS)
    style_ax(ax)

    axes[0, 3].set_visible(False)

    for idx, task in enumerate(TASK_HEADS):
        ax = axes[1, idx]
        train_key = f"train_macro_f1_{task}"
        val_key = f"val_macro_f1_{task}"
        if train_key in history and val_key in history:
            ax.plot(epochs, [a * 100 for a in history[train_key]],
                    color=colors["train"], label="训练", **line_kw)
            ax.plot(epochs, [a * 100 for a in history[val_key]],
                    color=colors["val"], label="验证", **line_kw)
        ax.set_xlabel("Epoch", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
        ax.set_ylabel("宏 F1 (%)", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
        ax.set_title(
            task_zh.get(task, task), fontsize=24, fontproperties=fp, pad=18, loc="left",
        )
        ax.legend(
            fontsize=_TRAIN_LEGEND_FS, prop=fp, framealpha=0.92,
            loc=_TASK_LEGEND_LOC.get(task, "best"),
        )
        ax.tick_params(labelsize=_TRAIN_TICK_FS)
        ax.set_ylim(0, 100)
        style_ax(ax)

    plt.tight_layout(pad=2.5)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"训练曲线已保存：{save_path}")


# ============================================================
# 记录环境信息
# ============================================================
def log_environment():
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"OS: {platform.platform()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ============================================================
# 单折训练
# ============================================================
def train_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    full_dataset: MultimodalFeatureDataset,
    device,
    loader_kw: dict,
) -> tuple:
    seed_everything(SEED + fold_idx)

    train_sampler = make_weighted_sampler(full_dataset, train_idx)
    train_loader  = DataLoader(Subset(full_dataset, train_idx), sampler=train_sampler, **loader_kw)
    val_loader    = DataLoader(Subset(full_dataset, val_idx),   shuffle=False, **loader_kw)

    # 与 dinov3 一致：WeightedRandomSampler 已逆频均衡 batch，Focal 不再带 class weight；四任务等权
    species_criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
    task_criteria = {
        task: FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
        for task in TASK_HEADS
    }
    task_loss_weights = {t: 1.0 for t in TASK_HEADS}
    logger.info(
        f"  [折{fold_idx+1}] 任务损失权重（等权）: " +
        ", ".join(f"{t}={w:.1f}" for t, w in task_loss_weights.items()),
    )

    model     = MultiBranchRoutingMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_warmup_cosine_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    history = {
        "train_loss": [], "val_loss": [],
        "train_balanced_macro_f1": [], "val_balanced_macro_f1": [], "lr": [],
    }
    for t in TASK_HEADS:
        history[f"train_macro_f1_{t}"] = []
        history[f"val_macro_f1_{t}"] = []

    best_metric = 0.0
    patience_count = 0
    best_state = None
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_ep = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc, tr_task_acc, tr_task_f1 = run_epoch(
            model, train_loader, species_criterion, task_criteria, optimizer, device,
            mixup_alpha=MIXUP_ALPHA, task_loss_weights=task_loss_weights,
        )
        val_loss, val_acc, val_task_acc, val_task_f1 = run_epoch(
            model, val_loader, species_criterion, task_criteria, None, device,
        )
        scheduler.step()

        train_balanced = float(np.mean([tr_task_f1[t] for t in TASK_HEADS]))
        val_balanced = float(np.mean([val_task_f1[t] for t in TASK_HEADS]))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_balanced_macro_f1"].append(train_balanced)
        history["val_balanced_macro_f1"].append(val_balanced)
        history["lr"].append(current_lr)
        for t in TASK_HEADS:
            history[f"train_macro_f1_{t}"].append(tr_task_f1.get(t, 0.0))
            history[f"val_macro_f1_{t}"].append(val_task_f1.get(t, 0.0))

        elapsed = time.time() - t_ep
        task_str = " | ".join(
            f"{t}:宏F1={val_task_f1.get(t, 0)*100:.1f}%"
            for t in TASK_HEADS
        )
        logger.info(
            f"[折{fold_idx+1}] Epoch [{epoch:>3d}/{NUM_EPOCHS}] "
            f"训练: loss={tr_loss:.4f} | "
            f"验证: loss={val_loss:.4f} balanced_macro_F1={val_balanced*100:.2f}% | "
            f"{task_str} | lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_balanced > best_metric + 1e-4:
            best_metric = val_balanced
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(
                f"  >>> 最优权重已更新 (balanced_macro_F1={val_balanced*100:.2f}%)",
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(
                    f"\n早停触发：验证 balanced_macro_F1 连续 {PATIENCE} 轮未改善，停止训练。",
                )
                break

    elapsed_total = time.time() - t_start
    logger.info(
        f"\n[折{fold_idx+1}] 训练完成：共 {len(history['train_loss'])} 轮，"
        f"最优 balanced_macro_F1={best_metric*100:.2f}%，用时 {elapsed_total/60:.1f} 分钟",
    )
    return best_state, history


# ============================================================
# 主流程
# ============================================================
def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("大创项目 LanguageBind 多分支路由 MLP 正式训练（学术级）")
    logger.info("=" * 60)
    log_environment()
    logger.info(f"计算设备：{device}")
    logger.info(f"特征来源：LanguageBind（Image + Audio-FT, ICLR 2024）")
    logger.info(f"输入维度：{INPUT_DIM}（LanguageBind projection_dim）")
    logger.info(f"超参数：epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LR}, "
                f"hidden={HIDDEN_DIM}, patience={PATIENCE}, "
                f"mixup_alpha={MIXUP_ALPHA}, focal_gamma={FOCAL_GAMMA}, "
                f"noise_std={NOISE_STD}, 少数侧噪声×{MINORITY_FEATURE_NOISE_MULT}, "
                f"n_folds={N_FOLDS}, tta_steps={TTA_STEPS}")
    logger.info(
        "早停与最优权重：验证集 balanced_macro_F1 = 四任务 macro-F1 算术平均（与 dinov3 一致思路）",
    )
    logger.info(
        "数据策略：仅逆频采样器 + Focal（无 class weight）+ 任务等权；"
        "特征噪声强化「猫图 / 狗音频」少数侧（与数据分布一致）",
    )
    logger.info("=" * 60)

    # ── 1. 加载数据 ─────────────────────────────────────────
    logger.info("\n[1/5] 加载 LanguageBind 特征数据...")
    full_dataset = MultimodalFeatureDataset(FEAT_DIR)

    # ── 2. 固定测试集，train+val 作为交叉验证池 ──────────────────
    logger.info("\n[2/5] 分层采样划分数据集...")
    train_idx, val_idx, test_idx = stratified_split(
        full_dataset, TRAIN_RATIO, VAL_RATIO, SEED
    )
    trainval_idx = np.concatenate([train_idx, val_idx])
    logger.info(f"  测试集={len(test_idx)} 固定，训练+验证池={len(trainval_idx)}")
    logger.info(f"  训练模式：{'%d 折交叉验证集成' % N_FOLDS if N_FOLDS > 1 else '单次训练'}")

    loader_kw   = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(Subset(full_dataset, test_idx), shuffle=False, **loader_kw)

    # ── 3. K 折训练 ─────────────────────────────────────────
    logger.info("\n[3/5] 开始训练...")
    fold_states   = []
    fold_histories = []

    if N_FOLDS > 1:
        skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        tv_labels = full_dataset.stratify_labels[trainval_idx]
        for fold_idx, (tr_sub, vl_sub) in enumerate(skf.split(trainval_idx, tv_labels)):
            fold_train = trainval_idx[tr_sub]
            fold_val   = trainval_idx[vl_sub]
            logger.info(f"\n{'─'*50}")
            logger.info(f"  第 {fold_idx+1}/{N_FOLDS} 折  训练={len(fold_train)}, 验证={len(fold_val)}")
            logger.info(f"{'─'*50}")
            state, history = train_fold(fold_idx, fold_train, fold_val, full_dataset, device, loader_kw)
            fold_states.append(state)
            fold_histories.append(history)
    else:
        state, history = train_fold(0, train_idx, val_idx, full_dataset, device, loader_kw)
        fold_states.append(state)
        fold_histories.append(history)

    # ── 4. 绘制训练曲线（最后一折）─────────────────────────────
    logger.info("\n[4/5] 绘制训练曲线...")
    fig_path = FIG_DIR / f"lb_train_curve_{TS}.png"
    plot_training_curves(fold_histories[-1], fig_path)

    # ── 5. 集成评估（K 折 × TTA）─────────────────────────────
    logger.info(f"\n[5/5] 集成评估（{N_FOLDS} 折 × TTA {TTA_STEPS} 步）...")
    cm_data, prob_bundle = full_evaluation_ensemble(
        fold_states, test_loader, device, full_dataset, TTA_STEPS, TTA_NOISE_STD
    )
    if cm_data:
        from languagebindtrain.visualization import plot_confusion_matrices_split

        cm_img = FIG_DIR / f"lb_confusion_split_image_{TS}.png"
        cm_aud = FIG_DIR / f"lb_confusion_split_audio_{TS}.png"
        plot_confusion_matrices_split(cm_data, cm_img, cm_aud)
        logger.info("混淆矩阵（按模态拆分）已保存：%s ； %s", cm_img, cm_aud)
        try:
            from languagebindtrain.mlp_interpretability import run_mlp_interpretability_suite

            run_mlp_interpretability_suite(
                cm_data,
                prob_bundle,
                full_dataset.task_class_names,
                FIG_DIR,
                TS,
                prefix="lb_",
            )
        except Exception as e:
            logger.warning("可解释性补充图（各类别 P/R/F1、ROC、置信度、Top 混淆）失败：%s", e)

    # ── 保存模型 ───────────────────────────────────────────
    model_path = MODEL_DIR / f"LB_MultiBranchMLP_{TS}.pkl"
    torch.save({
        "model_states": fold_states,
        "n_folds":      N_FOLDS,
        "backbone":     "LanguageBind",
        "hyperparams": {
            "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM,
            "seed": SEED, "lr": LR, "batch_size": BATCH_SIZE,
            "mixup_alpha": MIXUP_ALPHA, "focal_gamma": FOCAL_GAMMA,
            "noise_std": NOISE_STD,
            "minority_feature_noise_mult": MINORITY_FEATURE_NOISE_MULT,
            "species_loss_coef": 0.3,
            "task_loss_equal": True,
            "n_folds": N_FOLDS,
            "tta_steps": TTA_STEPS, "tta_noise_std": TTA_NOISE_STD,
        },
        "task_meta": TASK_META,
    }, model_path)
    logger.info(f"模型已保存（含 {N_FOLDS} 折权重）：{model_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"训练日志：{log_path}")
    logger.info(f"模型文件：{model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
