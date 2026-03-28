# -*- coding: utf-8 -*-
"""
大创项目：基于 DINOv3 ConvNeXt-Tiny 蒸馏模型的端到端猫狗图像情绪识别
训练脚本 (v4 — 终极修复版)

v4 修复要点：
  9. 消除 Scheduler 崩溃隐患：Epoch 0 将所有参数（含冻结的）一次性注册到
     optimizer + scheduler，不再动态 add_param_group。AdamW 对 grad=None 的
     参数自动跳过，解冻后无缝接管且继承当前 Cosine 学习率，彻底杜绝
     LambdaLR 内部 base_lrs/lr_lambdas 长度不匹配导致的 IndexError。
  10. CAT_TASK_WEIGHT=1：WeightedRandomSampler 已使猫:狗批次约 7:4，勿再 ×3 放大猫损失，
      否则 Backbone 梯度过度偏向少量猫图、狗分支易崩。

v3 修复要点：
  1. 推理端到端：测试时用预测 species 做硬路由，不再泄露 GT 标签
  2. 去三重过补偿：只保留 WeightedRandomSampler + FocalLoss(γ)，移除 class_weights
  3. 移除不确定性加权：task_log_vars 死代码；猫狗任务损失同权（见 CAT_TASK_WEIGHT）
  4. 移除 SupCon：Batch 小 + Mixup 与硬标签 SupCon 数学互斥
  5. 修复 Scheduler 重置：解冻后不再重建 scheduler，保持分类头动量
  6. 数据增强温和化：禁用 CutMix、降低 RandAugment/旋转/擦除强度
  7. 多尺度精简：只取最后两个 Stage 的 GeM 特征，避免底层纹理过拟合
  8. 解冻保守化：最多解冻 Stage 3+4，底层永远冻结

技术栈：
  - Backbone：DINOv3 ConvNeXt-Tiny（29M，从 ViT-7B 蒸馏，本地权重）
  - 特征聚合：高层 Stage (3,4) GeM Pooling + 拼接投影
  - 微调策略：渐进式逐 Stage 解冻（最多到 Stage 3+4）+ 分层学习率
  - 分类头：残差 MLP Head (3 层 + Skip Connection) + SE 注意力门控路由
  - 损失函数：Focal Loss + 标签平滑（猫狗任务等权，避免与层均衡采样叠加）
  - 采样策略：分层采样划分 + WeightedRandomSampler 过采样
  - 数据增强：轻量 RandAugment + Mixup + 多视角 TTA
  - 正则化：Dropout + 权重衰减 + EMA 权重滑动平均
  - 调度器：Warmup + Cosine Annealing
  - 评估：per-class P/R/F1 + 混淆矩阵 + K 折集成 + TTA

运行：
  conda activate d2l
  python train/3_dinov3_convnext_finetune.py

输入：  data/dog_emotion_cropped/ 和 data/cat_671_cropped/（原始图像目录）
输出：
  moxing/DINOv3_ConvNeXt_{时间戳}.pkl
  figure/dinov3_train_curve_{时间戳}.png
  figure/dinov3_confusion_matrix_{时间戳}.png
  txt/dinov3_train_log_{时间戳}.txt
"""

import os
import sys
import time
import copy
import math
import logging
import platform
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score)
from scipy import stats as sp_stats
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# 路径配置
# ============================================================
ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "moxing"
FIG_DIR   = ROOT / "figure"
TXT_DIR   = ROOT / "txt"
for d in [MODEL_DIR, FIG_DIR, TXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 时间戳与文件日志必须在 main() 内初始化：Windows 下 DataLoader 多进程会再次 import
# 本模块，若在模块顶层写 FileHandler，每个 worker 都会新建一份空白 txt。
TS = None
log_path = None


def setup_run_logging():
    """仅在主进程调用一次，生成单一训练日志文件。"""
    global TS, log_path
    TS = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = TXT_DIR / f"dinov3_train_log_{TS}.txt"
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
BATCH_SIZE    = 64
NUM_EPOCHS    = 80
PATIENCE      = 15

# 分层学习率（backbone 逐 Stage 衰减，保护底层预训练特征）
BACKBONE_LR   = 5e-6
BACKBONE_LR_DECAY = 0.6       # Stage3 的 LR = BACKBONE_LR × 0.6
HEAD_LR       = 3e-4
WEIGHT_DECAY  = 1e-3
WARMUP_EPOCHS = 5

# 渐进式解冻
UNFREEZE_SCHEDULE = {
    1:  "head_only",          # Epoch 1-9:  仅训练分类头
    10: "stage4",             # Epoch 10-19: 解冻 Stage 4
    20: "stage3_4",           # Epoch 20+:  解冻 Stage 3+4（底层永远冻结）
}

# 数据划分
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1

# Mixup 全程开启（α=0.4 更强混合抑制过拟合），CutMix 禁用
MIXUP_ALPHA   = 0.4
CUTMIX_ALPHA  = 0.0
CUTMIX_PROB   = 0.0

LABEL_SMOOTH  = 0.1
FOCAL_GAMMA   = 2.0

# 猫/狗任务损失倍率。须为 1.0：WeightedRandomSampler 已在 (物种×情绪) 层做逆频均衡，
# 猫 7 类 vs 狗 4 类会使批次中猫样本约 7:4；若再 >1 会双重放大猫分支梯度、扭曲 Backbone。
CAT_TASK_WEIGHT = 1.0

# EMA：warmup 期间从 0.99 线性升至目标值，避免早期参数剧烈变化时滞后
EMA_DECAY_START = 0.99
EMA_DECAY_END   = 0.999

# K 折与 TTA
N_FOLDS       = 1
TTA_STEPS     = 5

# 各分支 Dropout（猫分支数据少，过高 dropout 导致收敛抖动）
TASK_DROPOUTS = {
    "dog_img": 0.25,
    "cat_img": 0.35,
}

# DINOv3 模型配置
DINOV3_MODEL_NAME = str(
    ROOT / "pretrained_modelscope" / "facebook" / "dinov3-convnext-tiny-pretrain-lvd1689m"
)
INPUT_SIZE = 224

# ============================================================
# 可复现性
# ============================================================
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=False + benchmark=True：牺牲完全可复现性换取 cuDNN 自动选最快卷积算法
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        # Ampere 及更新架构上加速 matmul / 卷积，对精度影响通常可忽略
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# 标签映射
# ============================================================
DOG_IMG_CLASSES = ["angry", "happy", "relaxed", "sad"]
CAT_IMG_CLASSES = ["Angry", "Disgusted", "Happy", "Normal", "Sad", "Scared", "Surprised"]

TASK_META = {
    "dog_img": {"species": 0, "classes": DOG_IMG_CLASSES},
    "cat_img": {"species": 1, "classes": CAT_IMG_CLASSES},
}
TASK_HEADS = list(TASK_META.keys())


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

    def forward_mix(self, logits: torch.Tensor,
                    targets_a: torch.Tensor, targets_b: torch.Tensor,
                    lam: float, mix_mask: torch.Tensor) -> torch.Tensor:
        """CutMix/Mixup 场景下的 soft label focal loss"""
        n_classes = logits.shape[-1]
        safe_targets_b = targets_b.clamp(0, n_classes - 1)

        ce_a = F.cross_entropy(logits, targets_a, weight=self.weight,
                               reduction="none", label_smoothing=self.label_smoothing)
        ce_b = F.cross_entropy(logits, safe_targets_b, weight=self.weight,
                               reduction="none", label_smoothing=self.label_smoothing)
        pt_a = torch.exp(-ce_a)
        pt_b = torch.exp(-ce_b)
        focal_a = ((1 - pt_a) ** self.gamma) * ce_a
        focal_b = ((1 - pt_b) ** self.gamma) * ce_b
        focal = torch.where(mix_mask, lam * focal_a + (1.0 - lam) * focal_b, focal_a)
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ============================================================
# 自适应填充至正方形（避免形变）
# ============================================================
class PadToSquare:
    def __init__(self, fill=128):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        padding = (0, pad1, 0, pad2) if w > h else (pad1, 0, pad2, 0)
        return transforms.functional.pad(img, padding, fill=self.fill, padding_mode='constant')


# ============================================================
# 图像数据集
# ============================================================
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class PetImageDataset(Dataset):
    """
    统一加载猫狗图像数据集，为每个样本记录 (path, label, species, task_name)。
    训练/验证/测试通过 Subset 索引实现划分。
    """

    def __init__(self, data_dirs: dict, transform=None):
        """
        data_dirs: {"dog_img": (root_path, class_list), "cat_img": (root_path, class_list)}
        """
        self.samples: list[tuple[Path, int, int, str]] = []
        self.transform = transform
        self.task_class_names = {}
        self._task_name_to_id = {}

        skipped = 0
        for task_id, (task_name, (root, class_list)) in enumerate(data_dirs.items()):
            species = TASK_META[task_name]["species"]
            self.task_class_names[task_name] = class_list
            self._task_name_to_id[task_name] = task_id

            for label, cls in enumerate(class_list):
                cls_dir = root / cls
                if not cls_dir.exists():
                    logger.warning(f"类别目录不存在：{cls_dir}")
                    continue
                for p in cls_dir.iterdir():
                    if p.suffix.lower() not in IMG_EXTENSIONS:
                        continue
                    try:
                        with Image.open(p) as im:
                            im.verify()
                        self.samples.append((p, label, species, task_name))
                    except Exception:
                        skipped += 1

            count = sum(1 for s in self.samples if s[3] == task_name)
            logger.info(f"  {task_name}: 扫描到 {count} 张图像 ({len(class_list)} 类)")

        if skipped > 0:
            logger.warning(f"  共剔除 {skipped} 张损坏图像")

        self._labels = np.array([s[1] for s in self.samples], dtype=np.int64)
        self._species = np.array([s[2] for s in self.samples], dtype=np.int64)
        self._task_ids = np.array([self._task_name_to_id[s[3]] for s in self.samples], dtype=np.int64)
        self.stratify_labels = self._task_ids * 100 + self._labels

        logger.info(f"  数据集合并完毕：共 {len(self.samples)} 张图像")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, species, task_name = self.samples[idx]
        try:
            with Image.open(path) as img:
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (128, 128, 128))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
        except Exception as e:
            logger.warning(f"读取失败 {path}: {e}，返回零张量")
            img = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)

        return {
            "image":   img,
            "label":   torch.tensor(label, dtype=torch.long),
            "species": torch.tensor(species, dtype=torch.long),
            "task_id": torch.tensor(self._task_name_to_id[task_name], dtype=torch.long),
        }


# ============================================================
# Transform 安全封装（每个 Subset 持有独立 transform，避免多进程竞争）
# ============================================================
class TransformSubset(Dataset):
    """将 transform 绑定到 Subset 上，而非全局修改 dataset.transform"""
    def __init__(self, dataset: PetImageDataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label, species, task_name = self.dataset.samples[real_idx]
        try:
            with Image.open(path) as img:
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (128, 128, 128))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
        except Exception as e:
            logger.warning(f"读取失败 {path}: {e}，返回零张量")
            img = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)
        return {
            "image":   img,
            "label":   torch.tensor(label, dtype=torch.long),
            "species": torch.tensor(species, dtype=torch.long),
            "task_id": torch.tensor(self.dataset._task_name_to_id[task_name], dtype=torch.long),
        }


class CatAwareTransformSubset(Dataset):
    """
    训练集专用 Subset：对猫类样本（species==1）应用更强的数据增强策略，
    对狗类样本（species==0）使用标准增强策略，按物种自动路由。
    这样在不改变磁盘文件的前提下，从预处理层面扩充猫类的有效多样性，
    弥补猫类（~671张）与狗类（~4000张）之间的数量鸿沟。
    """
    def __init__(self, dataset: PetImageDataset, indices,
                 dog_transform, cat_transform):
        self.dataset = dataset
        self.indices = indices
        self.dog_transform = dog_transform
        self.cat_transform = cat_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label, species, task_name = self.dataset.samples[real_idx]
        transform = self.cat_transform if species == 1 else self.dog_transform
        try:
            with Image.open(path) as img:
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (128, 128, 128))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")
                if transform is not None:
                    img = transform(img)
        except Exception as e:
            logger.warning(f"读取失败 {path}: {e}，返回零张量")
            img = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)
        return {
            "image":   img,
            "label":   torch.tensor(label, dtype=torch.long),
            "species": torch.tensor(species, dtype=torch.long),
            "task_id": torch.tensor(self.dataset._task_name_to_id[task_name], dtype=torch.long),
        }


# ============================================================
# 数据增强 (v2: 更强、更科学)
# ============================================================
def get_train_transform(processor_mean, processor_std):
    return transforms.Compose([
        PadToSquare(fill=128),
        transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor_mean, std=processor_std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])


def get_cat_train_transform(processor_mean, processor_std):
    """
    针对猫类（仅 671 张）的强增强策略。
    相比狗类增强，加大了旋转范围、色彩扰动、RandAugment 强度，
    并额外引入垂直翻转与透视变换，从 671 张中最大化挖掘样本多样性，
    缓解猫类因样本量极少（狗类 1/6）导致的过拟合风险。
    """
    return transforms.Compose([
        PadToSquare(fill=128),
        transforms.Resize((INPUT_SIZE + 48, INPUT_SIZE + 48),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=8),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.35, hue=0.08),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandAugment(num_ops=2, magnitude=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor_mean, std=processor_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20)),
    ])


def get_val_transform(processor_mean, processor_std):
    return transforms.Compose([
        PadToSquare(fill=128),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor_mean, std=processor_std),
    ])


def get_tta_transforms(processor_mean, processor_std):
    """TTA: 5 种确定性视角变换 + 2 种组合"""
    base = [
        PadToSquare(fill=128),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=processor_mean, std=processor_std),
    ]

    tta_list = [
        transforms.Compose(base + normalize),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + normalize),
        transforms.Compose(base + [transforms.RandomAffine(degrees=10, scale=(0.95, 1.05))] + normalize),
        transforms.Compose(base + [transforms.ColorJitter(brightness=0.15, contrast=0.15)] + normalize),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0),
                                    transforms.RandomAffine(degrees=8, scale=(0.93, 1.07))] + normalize),
    ]
    return tta_list


# ============================================================
# GeM Pooling (Generalized Mean Pooling)
# ============================================================
class GeM(nn.Module):
    """可学习的广义均值池化，p=1 退化为平均池化，p→∞ 退化为最大池化"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


# ============================================================
# SE (Squeeze-and-Excitation) 门控路由
# ============================================================
class SEGate(nn.Module):
    """通道注意力门控，用于物种路由前的特征重校准"""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 32)
        self.gate = nn.Sequential(
            nn.Linear(dim, mid),
            nn.GELU(),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


# ============================================================
# 残差 MLP 分类头 (3 层 + Skip Connection)
# ============================================================
class ResidualTaskHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        mid = max(in_dim // 2, 128)
        self.fc1 = nn.Linear(in_dim, mid)
        self.ln1 = nn.LayerNorm(mid)
        self.fc2 = nn.Linear(mid, mid)
        self.ln2 = nn.LayerNorm(mid)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(mid, num_classes)
        self.skip_proj = nn.Linear(in_dim, mid) if in_dim != mid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(x)
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = F.gelu(self.ln2(self.fc2(h)))
        # 仅对变换分支 Dropout，skip 恒等通路不被掩码打断，利于梯度与稳定收敛
        h = self.drop(h) + skip
        return self.classifier(h)


# ============================================================
# DINOv3 ConvNeXt-Tiny 多分支分类模型 (v2 精进版)
# ============================================================
class DINOv3MultibranchModel(nn.Module):
    """
    结构：
      Image → DINOv3 ConvNeXt-Tiny backbone
        → 高层 Stage (3,4) 特征 + GeM Pooling → 拼接 → 投影
        → SE 门控注意力 → 共享表征 (384-d)
        → head_species (2 类, 用于端到端硬路由)
        → head_dog_img (残差 MLP, 4 类)
        → head_cat_img (残差 MLP, 7 类)
    """

    def __init__(self, model_name: str = DINOV3_MODEL_NAME, hidden_dim: int = 384):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        all_dims = list(self.backbone.config.hidden_sizes)
        # 只取最后两个 Stage（高层语义特征），底层纹理在小样本下引入过拟合风险
        self.use_stage_indices = list(range(max(0, len(all_dims) - 2), len(all_dims)))
        self.stage_dims = [all_dims[i] for i in self.use_stage_indices]
        total_dim = sum(self.stage_dims)
        logger.info(f"  DINOv3 所有 Stage 维度: {all_dims}，使用 Stage {self.use_stage_indices}，拼接后: {total_dim}")

        self.stage_gems = nn.ModuleList([GeM(p=3.0) for _ in self.stage_dims])

        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.se_gate = SEGate(hidden_dim, reduction=4)

        self.head_species = nn.Linear(hidden_dim, 2)
        self.head_dog_img = ResidualTaskHead(hidden_dim, len(DOG_IMG_CLASSES), TASK_DROPOUTS["dog_img"])
        self.head_cat_img = ResidualTaskHead(hidden_dim, len(CAT_IMG_CLASSES), TASK_DROPOUTS["cat_img"])

    def _aggregate_features(self, hidden_states: tuple) -> torch.Tensor:
        # hidden_states[0] 是 embedding，hidden_states[1:] 是各 Stage 输出
        all_stage_feats = hidden_states[1:]
        stage_features = []
        for idx, gem in zip(self.use_stage_indices, self.stage_gems):
            feat = all_stage_feats[idx]
            if feat.dim() == 4:
                pooled = gem(feat)
            else:
                pooled = feat.mean(dim=1)
            stage_features.append(pooled)
        return torch.cat(stage_features, dim=-1)

    def forward(self, pixel_values: torch.Tensor,
                species: torch.Tensor = None,
                return_hidden: bool = False) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        agg = self._aggregate_features(outputs.hidden_states)
        h = self.projection(agg)
        h = self.se_gate(h)

        species_logits = self.head_species(h)
        result = {"species": species_logits}

        # 训练时使用 GT species 做路由（teacher forcing），推理时使用预测 species 做硬路由
        if species is not None:
            route = species
        else:
            route = species_logits.argmax(dim=1)

        dog_mask = (route == 0)
        cat_mask = (route == 1)
        dog_logits = h.new_zeros(h.shape[0], len(DOG_IMG_CLASSES))
        cat_logits = h.new_zeros(h.shape[0], len(CAT_IMG_CLASSES))
        if dog_mask.any():
            dog_logits[dog_mask] = self.head_dog_img(h[dog_mask]).to(dog_logits.dtype)
        if cat_mask.any():
            cat_logits[cat_mask] = self.head_cat_img(h[cat_mask]).to(cat_logits.dtype)
        result["dog_img"] = dog_logits
        result["cat_img"] = cat_logits

        if return_hidden:
            result["hidden"] = h
        return result

    def get_param_groups(self, backbone_lr: float, head_lr: float,
                         weight_decay: float, lr_decay: float = 0.6):
        """逐 Stage 分层衰减学习率 + LayerNorm/Bias 不施加 weight decay。
        Epoch 0 即把所有参数（含冻结的）一次性注册到 optimizer，
        AdamW 对 requires_grad=False 的参数自动跳过。"""
        def _is_no_decay(name: str) -> bool:
            return ("norm" in name.lower() or "ln" in name.lower()
                    or "layernorm" in name.lower() or name.endswith(".bias"))

        if hasattr(self.backbone, 'stages'):
            stages = self.backbone.stages
        elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'stages'):
            stages = self.backbone.encoder.stages
        else:
            stages = None

        groups = []
        backbone_param_ids = set()

        if stages is not None:
            n_stages = len(stages)
            for si in range(n_stages):
                depth_from_top = n_stages - 1 - si
                stage_lr = backbone_lr * (lr_decay ** depth_from_top)
                wd_params, no_wd_params = [], []
                for n, p in stages[si].named_parameters():
                    backbone_param_ids.add(id(p))
                    if _is_no_decay(n):
                        no_wd_params.append(p)
                    else:
                        wd_params.append(p)
                if wd_params:
                    groups.append({"params": wd_params, "lr": stage_lr,
                                   "weight_decay": weight_decay})
                if no_wd_params:
                    groups.append({"params": no_wd_params, "lr": stage_lr,
                                   "weight_decay": 0.0})

            other_bb_wd, other_bb_nowd = [], []
            for n, p in self.backbone.named_parameters():
                if id(p) not in backbone_param_ids:
                    backbone_param_ids.add(id(p))
                    if _is_no_decay(n):
                        other_bb_nowd.append(p)
                    else:
                        other_bb_wd.append(p)
            min_lr = backbone_lr * (lr_decay ** (n_stages - 1))
            if other_bb_wd:
                groups.append({"params": other_bb_wd, "lr": min_lr,
                               "weight_decay": weight_decay})
            if other_bb_nowd:
                groups.append({"params": other_bb_nowd, "lr": min_lr,
                               "weight_decay": 0.0})
        else:
            backbone_param_ids = set(id(p) for p in self.backbone.parameters())
            bb_wd = [p for n, p in self.backbone.named_parameters() if not _is_no_decay(n)]
            bb_nowd = [p for n, p in self.backbone.named_parameters() if _is_no_decay(n)]
            if bb_wd:
                groups.append({"params": bb_wd, "lr": backbone_lr,
                               "weight_decay": weight_decay})
            if bb_nowd:
                groups.append({"params": bb_nowd, "lr": backbone_lr,
                               "weight_decay": 0.0})

        head_wd, head_nowd = [], []
        for n, p in self.named_parameters():
            if id(p) not in backbone_param_ids:
                if _is_no_decay(n):
                    head_nowd.append(p)
                else:
                    head_wd.append(p)
        if head_wd:
            groups.append({"params": head_wd, "lr": head_lr,
                           "weight_decay": weight_decay * 0.1})
        if head_nowd:
            groups.append({"params": head_nowd, "lr": head_lr,
                           "weight_decay": 0.0})

        return groups

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("  Backbone 已冻结（全部 Stage）")

    def progressive_unfreeze(self, stage_name: str):
        """渐进式解冻 ConvNeXt 的各 Stage"""
        if hasattr(self.backbone, 'stages'):
            stages = self.backbone.stages
        elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'stages'):
            stages = self.backbone.encoder.stages
        else:
            stages = None
        if stages is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
            logger.info(f"  Backbone 已全部解冻（无法识别 Stage 结构）")
            return

        n_stages = len(stages)
        if stage_name == "stage4":
            for p in stages[n_stages - 1].parameters():
                p.requires_grad = True
            logger.info(f"  已解冻 Stage {n_stages}")
        elif stage_name == "stage3_4":
            for i in range(max(0, n_stages - 2), n_stages):
                for p in stages[i].parameters():
                    p.requires_grad = True
            logger.info(f"  已解冻 Stage {n_stages-1}-{n_stages}")
        elif stage_name == "all":
            for p in self.backbone.parameters():
                p.requires_grad = True
            logger.info(f"  Backbone 已全部解冻（{n_stages} 个 Stage）")


# ============================================================
# 分层采样
# ============================================================
def stratified_split(dataset: PetImageDataset, train_ratio, val_ratio, seed):
    labels = dataset.stratify_labels
    n = len(labels)
    test_ratio = 1.0 - train_ratio - val_ratio

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), labels))

    trainval_labels = labels[trainval_idx]
    relative_val = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_sub, val_sub = next(sss2.split(np.zeros(len(trainval_idx)), trainval_labels))

    return trainval_idx[train_sub], trainval_idx[val_sub], test_idx


def make_weighted_sampler(dataset: PetImageDataset, indices: np.ndarray) -> WeightedRandomSampler:
    strat = dataset.stratify_labels[indices]
    counter = Counter(strat)
    weights = np.array([1.0 / counter[s] for s in strat], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)


# ============================================================
# 任务分支掩码（按物种路由）
# ============================================================
TASK_MASKS = {
    "dog_img": lambda s: (s == 0),
    "cat_img": lambda s: (s == 1),
}


# ============================================================
# 损失计算
# ============================================================
def compute_loss(outputs, batch, species_criterion, task_criteria):
    species = batch["species"]
    labels  = batch["label"]

    mix_lam = batch.get("mix_lam", None)
    mix_labels_b = batch.get("mix_labels_b", None)
    mix_mask = batch.get("mix_mask", None)

    loss_species = species_criterion(outputs["species"], species)
    loss_total = 0.3 * loss_species
    loss_dict = {"species": loss_species.item()}

    for task in TASK_HEADS:
        mask = TASK_MASKS[task](species)
        if not mask.any():
            continue
        logits_t = outputs[task][mask]
        labels_t = labels[mask]

        if mix_lam is not None and mix_labels_b is not None:
            labels_b_t = mix_labels_b[mask]
            mix_mask_t = mix_mask[mask] if mix_mask is not None else torch.ones_like(labels_t, dtype=torch.bool)
            n_cls = logits_t.shape[-1]
            oob = (labels_b_t < 0) | (labels_b_t >= n_cls)
            if oob.any():
                labels_b_t = labels_b_t.clone()
                labels_b_t[oob] = labels_t[oob]
                mix_mask_t = mix_mask_t.clone()
                mix_mask_t[oob] = False
            t_loss = task_criteria[task].forward_mix(logits_t, labels_t, labels_b_t, mix_lam, mix_mask_t)
        else:
            t_loss = task_criteria[task](logits_t, labels_t)

        # 与采样器一致：层均衡后不再额外放大猫任务（见文件顶部 CAT_TASK_WEIGHT）
        w = CAT_TASK_WEIGHT if task == "cat_img" else 1.0
        t_loss = w * t_loss

        loss_total = loss_total + t_loss
        loss_dict[task] = t_loss.item()

    return loss_total, loss_dict


def compute_accuracy_detailed(outputs, batch):
    species = batch["species"]
    labels  = batch["label"]
    result = {}
    for task in TASK_HEADS:
        mask = TASK_MASKS[task](species)
        if mask.any():
            preds = outputs[task][mask].argmax(dim=1)
            correct = (preds == labels[mask]).sum().item()
            count = mask.sum().item()
            result[task] = (correct, count)
    return result


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================
class EMA:
    """模型参数指数滑动平均，支持 warmup decay schedule"""
    def __init__(self, model: nn.Module, decay_start: float = 0.99,
                 decay_end: float = 0.999, warmup_steps: int = 0):
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.warmup_steps = max(warmup_steps, 1)
        self._step = 0
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    def _get_decay(self) -> float:
        if self._step >= self.warmup_steps:
            return self.decay_end
        ratio = self._step / self.warmup_steps
        return self.decay_start + (self.decay_end - self.decay_start) * ratio

    @torch.no_grad()
    def update(self, model: nn.Module):
        decay = self._get_decay()
        self._step += 1
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(decay).add_(v, alpha=1.0 - decay)
            else:
                self.shadow[k].copy_(v)

    def apply_shadow(self, model: nn.Module):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        model.load_state_dict(self.backup)
        self.backup = {}

    def state_dict(self):
        return self.shadow


# ============================================================
# Warmup + CosineAnnealing with Warm Restarts
# ============================================================
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                                min_lr_ratio=0.01, restart_period=0):
    """
    restart_period > 0 时启用 warm restarts（每 restart_period 轮重启一次余弦）
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        post = epoch - warmup_epochs
        remain = total_epochs - warmup_epochs
        if restart_period > 0:
            cycle_pos = post % restart_period
            cycle_len = restart_period
        else:
            cycle_pos = post
            cycle_len = remain
        progress = cycle_pos / max(cycle_len, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# CutMix / Mixup 混合增强
# ============================================================
def _rand_bbox(H, W, lam):
    cut_rat = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    return y1, y2, x1, x2


def mixed_augment_batch(batch: dict, mixup_alpha: float,
                        cutmix_alpha: float, cutmix_prob: float) -> dict:
    """随机选择 CutMix 或 Mixup，仅在同物种样本间混合，并返回 soft label 信息"""
    images   = batch["image"]
    task_ids = batch["task_id"]
    labels   = batch["label"]
    B, C, H, W = images.shape

    perm = torch.randperm(B, device=images.device)
    same_task = (task_ids == task_ids[perm])

    use_cutmix = np.random.rand() < cutmix_prob

    if use_cutmix and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        y1, y2, x1, x2 = _rand_bbox(H, W, lam)
        mixed = images.clone()
        mixed[same_task, :, y1:y2, x1:x2] = images[perm][same_task, :, y1:y2, x1:x2]
        lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    elif mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        lam = max(lam, 1 - lam)
        mixed = images.clone()
        mixed[same_task] = lam * images[same_task] + (1 - lam) * images[perm][same_task]
        lam_actual = lam
    else:
        return batch

    batch_out = {k: v for k, v in batch.items()}
    batch_out["image"] = mixed
    batch_out["mix_lam"] = lam_actual
    batch_out["mix_labels_b"] = labels[perm]
    batch_out["mix_mask"] = same_task
    return batch_out


# ============================================================
# 训练 / 验证一轮
# ============================================================
def run_epoch(model, loader, species_criterion, task_criteria,
              optimizer=None, device="cpu",
              scaler=None, epoch=0, ema=None,
              use_augmix=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_count = 0.0, 0
    task_correct = {t: 0 for t in TASK_HEADS}
    task_count   = {t: 0 for t in TASK_HEADS}
    task_all_preds  = {t: [] for t in TASK_HEADS}
    task_all_labels = {t: [] for t in TASK_HEADS}
    amp_device = device.type if device.type == "cuda" else "cpu"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            if device.type == "cuda":
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["image"].shape[0]

            if is_train and use_augmix:
                batch = mixed_augment_batch(batch, MIXUP_ALPHA, CUTMIX_ALPHA, CUTMIX_PROB)

            if scaler is not None and is_train:
                with torch.amp.autocast(amp_device):
                    outputs = model(batch["image"], species=batch["species"])
                    loss, ld = compute_loss(outputs, batch, species_criterion, task_criteria)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch["image"], species=batch["species"])
                loss, ld = compute_loss(outputs, batch, species_criterion, task_criteria)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            if is_train and ema is not None:
                ema.update(model)

            total_loss  += loss.item() * B
            total_count += B

            species = batch["species"]
            labels  = batch["label"]
            for task in TASK_HEADS:
                mask = TASK_MASKS[task](species)
                if mask.any():
                    preds = outputs[task][mask].argmax(dim=1)
                    task_all_preds[task].append(preds.detach().cpu())
                    task_all_labels[task].append(labels[mask].detach().cpu())
                    task_correct[task] += (preds == labels[mask]).sum().item()
                    task_count[task] += mask.sum().item()

    avg_loss = total_loss / max(total_count, 1)
    task_acc = {t: task_correct[t] / task_count[t] if task_count[t] > 0 else 0.0
                for t in TASK_HEADS}
    overall_correct = sum(task_correct.values())
    overall_count = sum(task_count.values())
    overall_acc = overall_correct / max(overall_count, 1)

    task_f1 = {}
    for t in TASK_HEADS:
        if task_all_preds[t]:
            all_p = torch.cat(task_all_preds[t]).numpy()
            all_l = torch.cat(task_all_labels[t]).numpy()
            _, _, f1, _ = precision_recall_fscore_support(all_l, all_p, average="macro",
                                                          zero_division=0)
            task_f1[t] = f1
        else:
            task_f1[t] = 0.0

    return avg_loss, overall_acc, task_acc, task_f1


# ============================================================
# TTA 评估
# ============================================================
@torch.no_grad()
def predict_tta(model, dataset, test_indices, tta_transforms, device, batch_size=32):
    """多视角 TTA：对同一张图使用不同增强，softmax 概率取均值
    
    端到端推理：先用模型预测 species，再根据预测结果路由到对应情绪头
    """
    model.eval()

    task_probs = {t: None for t in TASK_HEADS}
    task_targets = {t: None for t in TASK_HEADS}
    _n_workers = 4 if os.name == "nt" else 8
    amp_device = device.type if device.type == "cuda" else "cpu"

    for tta_idx, tfm in enumerate(tta_transforms):
        tta_ds = TransformSubset(dataset, test_indices, tfm)
        loader = DataLoader(
            tta_ds, batch_size=batch_size, shuffle=False,
            num_workers=_n_workers, pin_memory=(device.type == "cuda"),
        )

        all_pred_species, all_gt_species, all_labels = [], [], []
        all_logits = {t: [] for t in TASK_HEADS}

        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(amp_device):
                # species=None → 模型内部用预测 species 做硬路由
                outputs = model(batch["image"])

            pred_sp = outputs["species"].argmax(dim=1)
            all_pred_species.append(pred_sp.cpu())
            all_gt_species.append(batch["species"].cpu())
            all_labels.append(batch["label"].cpu())
            for task in TASK_HEADS:
                all_logits[task].append(outputs[task].cpu())

        all_pred_species = torch.cat(all_pred_species)
        all_gt_species   = torch.cat(all_gt_species)
        all_labels       = torch.cat(all_labels)

        for task in TASK_HEADS:
            logits = torch.cat(all_logits[task])
            # 用 GT species 筛选目标样本（确定哪些样本属于该任务）
            gt_mask = TASK_MASKS[task](all_gt_species)
            if not gt_mask.any():
                continue

            probs = F.softmax(logits[gt_mask], dim=-1).numpy()
            tgts  = all_labels[gt_mask].numpy()

            if task_probs[task] is None:
                task_probs[task] = probs
                task_targets[task] = tgts
            else:
                task_probs[task] += probs

    n_tta = len(tta_transforms)
    for task in TASK_HEADS:
        if task_probs[task] is not None:
            task_probs[task] /= n_tta

    # 计算物种分类准确率（端到端路由的可靠性指标）
    if len(all_pred_species) > 0 and len(all_gt_species) > 0:
        sp_acc = (all_pred_species == all_gt_species).float().mean().item()
        logger.info(f"  物种分类准确率 (端到端路由): {sp_acc*100:.2f}%")

    return task_probs, task_targets


# ============================================================
# 完整评估 + 报告
# ============================================================
def _wilson_confidence_interval(acc: float, n: int, z: float = 1.96) -> tuple:
    """Wilson 置信区间，比 Wald 区间在小样本下更准确"""
    if n == 0:
        return 0.0, 0.0
    p_hat = acc
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def full_evaluation(task_probs, task_targets, dataset):
    report_lines = []
    cm_data = {}

    for task in TASK_HEADS:
        if task_probs[task] is None:
            continue
        preds = task_probs[task].argmax(axis=1)
        targets = task_targets[task]
        probs = task_probs[task]
        class_names = dataset.task_class_names[task]
        n_cls = len(class_names)

        p, r, f1, sup = precision_recall_fscore_support(
            targets, preds, average=None,
            labels=list(range(n_cls)), zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0,
        )
        weighted_f1 = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0,
        )[2]
        acc = (preds == targets).mean()
        acc_pct = acc * 100
        ci_lo, ci_hi = _wilson_confidence_interval(acc, len(targets))

        try:
            if n_cls == 2:
                auc = roc_auc_score(targets, probs[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                targets_bin = label_binarize(targets, classes=list(range(n_cls)))
                auc = roc_auc_score(targets_bin, probs, average="macro", multi_class="ovr")
        except Exception:
            auc = float("nan")

        report_lines.append(f"\n{'═' * 65}")
        report_lines.append(f"任务: {task}  Accuracy={acc_pct:.2f}% "
                            f"(95% CI: [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%])  "
                            f"AUC-ROC={auc:.4f}")
        report_lines.append(f"{'─' * 65}")
        report_lines.append(f"{'类别':>14s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'支持数':>6s}")
        for j, cls in enumerate(class_names):
            report_lines.append(
                f"  {cls:>12s}  {p[j]*100:>8.2f}%  {r[j]*100:>8.2f}%  {f1[j]*100:>8.2f}%  {int(sup[j]):>6d}"
            )
        report_lines.append(f"{'─' * 65}")
        report_lines.append(f"  {'Macro Avg':>12s}  {macro_p*100:>8.2f}%  {macro_r*100:>8.2f}%  {macro_f1*100:>8.2f}%")
        report_lines.append(f"  {'Weighted F1':>12s}  {'':>9s}  {'':>9s}  {weighted_f1*100:>8.2f}%")

        cm = confusion_matrix(targets, preds, labels=list(range(n_cls)))
        cm_data[task] = (cm, class_names)

    report = "\n".join(report_lines)
    logger.info(report)
    return cm_data, task_probs, task_targets


# ============================================================
# K 折集成评估
# ============================================================
def ensemble_evaluation(fold_states, dataset, test_indices,
                        tta_transforms, device):
    fold_probs = {t: [] for t in TASK_HEADS}
    targets_ref = {t: None for t in TASK_HEADS}

    for fold_idx, state in enumerate(fold_states):
        model = DINOv3MultibranchModel()
        model.load_state_dict(state, strict=False)
        model = model.to(device)
        model.eval()

        probs, tgts = predict_tta(model, dataset, test_indices,
                                   tta_transforms, device, BATCH_SIZE)
        for task in TASK_HEADS:
            if probs[task] is not None:
                fold_probs[task].append(probs[task])
                if targets_ref[task] is None:
                    targets_ref[task] = tgts[task]
        logger.info(f"  折 {fold_idx + 1}/{len(fold_states)} TTA 推理完成")

        del model
        torch.cuda.empty_cache()

    avg_probs = {}
    for task in TASK_HEADS:
        if fold_probs[task]:
            avg_probs[task] = np.mean(fold_probs[task], axis=0)
        else:
            avg_probs[task] = None

    return full_evaluation(avg_probs, targets_ref, dataset)
# ============================================================
def _get_zh_font():
    zh_fonts = [f.fname for f in fm.fontManager.ttflist
                if any(kw in f.name for kw in
                       ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong"])]
    if zh_fonts:
        return fm.FontProperties(fname=zh_fonts[0])
    return fm.FontProperties()


# 与 DINO3train/visualization 对齐
_TRAIN_LEGEND_FS = 22
_TRAIN_TICK_FS = 20
_TRAIN_AXIS_FS = 20
# 子图标题须用 FontProperties.set_size：仅用 fontsize= 会被 fname 载入的默认字号覆盖
_TRAIN_TASK_TITLE_FS = 28
_TASK_LEGEND_LOC = {"dog_img": "upper left", "cat_img": "upper right"}
_CM_TITLE_FS = 20
_CM_AXIS_NAME_FS = 17
_CM_CLASS_TICK_FS = 16


def plot_training_curves(history: dict, save_path: Path):
    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor("white")
    line_kw = dict(linewidth=2.2, marker="o", markersize=3)
    colors = {"train": "#2E86AB", "val": "#E84855"}
    task_zh = {"dog_img": "狗图像情绪", "cat_img": "猫图像情绪"}
    task_title_fp = fp.copy()
    task_title_fp.set_size(_TRAIN_TASK_TITLE_FS)

    def style_ax(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.2)

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color=colors["train"], label="训练", **line_kw)
    ax.plot(epochs, history["val_loss"], color=colors["val"], label="验证", **line_kw)
    ax.set_xlabel("Epoch", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.set_ylabel("Loss", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.legend(fontsize=_TRAIN_LEGEND_FS, prop=fp, framealpha=0.92)
    ax.tick_params(labelsize=_TRAIN_TICK_FS)
    style_ax(ax)

    ax = axes[0, 1]
    ax.plot(epochs, [a*100 for a in history["train_acc"]], color=colors["train"], label="训练", **line_kw)
    ax.plot(epochs, [a*100 for a in history["val_acc"]], color=colors["val"], label="验证", **line_kw)
    ax.set_xlabel("Epoch", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
    ax.set_ylabel("准确率 (%)", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
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

    for idx, task in enumerate(TASK_HEADS):
        ax = axes[1, idx]
        tr_key = f"train_acc_{task}"
        vl_key = f"val_acc_{task}"
        if tr_key in history:
            ax.plot(epochs, [a*100 for a in history[tr_key]],
                    color=colors["train"], label="训练", **line_kw)
            ax.plot(epochs, [a*100 for a in history[vl_key]],
                    color=colors["val"], label="验证", **line_kw)
        ax.set_xlabel("Epoch", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
        ax.set_ylabel("准确率 (%)", fontsize=_TRAIN_AXIS_FS, fontproperties=fp)
        ax.set_title(
            task_zh.get(task, task), fontproperties=task_title_fp, pad=18, loc="left",
        )
        ax.legend(
            fontsize=_TRAIN_LEGEND_FS, prop=fp, framealpha=0.92,
            loc=_TASK_LEGEND_LOC.get(task, "best"),
        )
        ax.tick_params(labelsize=_TRAIN_TICK_FS)
        ax.set_ylim(0, 118)
        style_ax(ax)

    axes[1, 2].set_visible(False)

    plt.tight_layout(pad=2.5)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"训练曲线已保存：{save_path}")


def plot_roc_curves(task_probs: dict, task_targets: dict,
                    class_names_map: dict, save_path: Path):
    """
    为每个任务各绘制一条微平均（micro-average）ROC 曲线，图例仅显示微平均 AUC 值。
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    n_tasks = sum(1 for t in TASK_HEADS if task_probs.get(t) is not None)
    if n_tasks == 0:
        return

    _roc_title_fs = 26
    _roc_axis_fs  = 22
    _roc_legend_fs = 20
    _roc_tick_fs  = 20
    title_fp  = fp.copy(); title_fp.set_size(_roc_title_fs)
    axis_fp   = fp.copy(); axis_fp.set_size(_roc_axis_fs)
    leg_fp    = fp.copy(); leg_fp.set_size(_roc_legend_fs)

    task_zh    = {"dog_img": "狗图像情绪", "cat_img": "猫图像情绪"}
    task_color = {"dog_img": "#2E86AB",    "cat_img": "#E84855"}

    fig, axes = plt.subplots(1, n_tasks, figsize=(10 * n_tasks, 9))
    fig.patch.set_facecolor("white")
    if n_tasks == 1:
        axes = [axes]

    ax_idx = 0
    for task in TASK_HEADS:
        if task_probs.get(task) is None:
            continue
        ax = axes[ax_idx]; ax_idx += 1
        ax.set_facecolor("white")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)

        probs   = task_probs[task]   # (N, n_cls)
        targets = task_targets[task] # (N,)
        n_cls   = len(class_names_map[task])

        targets_bin = label_binarize(targets, classes=list(range(n_cls)))
        if n_cls == 2:
            targets_bin = np.hstack([1 - targets_bin, targets_bin])

        # 微平均：将所有类别的预测拼成一个长向量后计算 ROC
        fpr_micro, tpr_micro, _ = roc_curve(
            targets_bin.ravel(), probs.ravel()
        )
        auc_micro = auc(fpr_micro, tpr_micro)

        ax.plot(fpr_micro, tpr_micro,
                color=task_color.get(task, "#333333"),
                linewidth=2.8,
                label=f"微平均 AUC = {auc_micro:.3f}")

        # 对角参考线
        ax.plot([0, 1], [0, 1], color="#AAAAAA", linewidth=1.5,
                linestyle=":", zorder=0)

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.05)
        ax.set_xlabel("假阳性率 (FPR)", fontproperties=axis_fp)
        ax.set_ylabel("真阳性率 (TPR)", fontproperties=axis_fp)
        ax.set_title(task_zh.get(task, task) + " ROC 曲线",
                     fontproperties=title_fp, pad=18)
        ax.tick_params(labelsize=_roc_tick_fs)
        ax.legend(prop=leg_fp, loc="lower right", framealpha=0.92)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"ROC 曲线已保存：{save_path}")


def plot_confusion_matrices(cm_data: dict, save_path: Path):
    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False
    task_zh = {"dog_img": "狗图像情绪", "cat_img": "猫图像情绪"}
    title_fp = fp.copy()
    title_fp.set_size(_CM_TITLE_FS)
    axis_name_fp = fp.copy()
    axis_name_fp.set_size(_CM_AXIS_NAME_FS)

    n_tasks = len(cm_data)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6.5))
    fig.patch.set_facecolor("white")
    if n_tasks == 1:
        axes = [axes]

    for ax, (task, (cm, class_names)) in zip(axes, cm_data.items()):
        cm_pct = cm.astype(np.float64)
        row_sums = np.maximum(cm_pct.sum(axis=1, keepdims=True), 1)
        cm_pct = cm_pct / row_sums * 100

        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        n = len(class_names)
        # 类别多时格子小，需单独缩小格内数字以免溢出
        if n <= 4:
            cell_fs = 12
        else:
            cell_fs = max(7, 10 - (n - 4))
        for i in range(n):
            for j in range(n):
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                        ha="center", va="center",
                        fontsize=cell_fs, color=color, fontproperties=fp)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(
            class_names, fontsize=_CM_CLASS_TICK_FS, fontproperties=fp, rotation=45, ha="right",
        )
        ax.set_yticklabels(class_names, fontsize=_CM_CLASS_TICK_FS, fontproperties=fp)
        ax.set_xlabel("预测", fontproperties=axis_name_fp)
        ax.set_ylabel("真实", fontproperties=axis_name_fp)
        ax.set_title(task_zh.get(task, task), fontproperties=title_fp, pad=22)
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_linewidth(1.2)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"混淆矩阵已保存：{save_path}")


# ============================================================
# t-SNE 特征空间可视化
# ============================================================
@torch.no_grad()
def plot_tsne_features(model_state, dataset, test_indices, val_transform,
                       device, save_path: Path, max_samples: int = 800):
    from sklearn.manifold import TSNE

    model = DINOv3MultibranchModel().to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    test_ds = TransformSubset(dataset, test_indices, val_transform)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))

    all_hidden, all_labels, all_species = [], [], []
    amp_device = device.type if device.type == "cuda" else "cpu"
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(amp_device):
            outputs = model(batch["image"], return_hidden=True)
        all_hidden.append(outputs["hidden"].cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        all_species.append(batch["species"].cpu().numpy())

    feats = np.concatenate(all_hidden)
    labels = np.concatenate(all_labels)
    species = np.concatenate(all_species)

    if len(feats) > max_samples:
        idx = np.random.choice(len(feats), max_samples, replace=False)
        feats, labels, species = feats[idx], labels[idx], species[idx]

    tsne = TSNE(n_components=2, perplexity=min(30, len(feats) - 1),
                random_state=SEED, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(feats)

    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False
    task_titles_zh = {"dog_img": "狗图像情绪特征分布", "cat_img": "猫图像情绪特征分布"}
    _tsne_title_fs = 36
    _tsne_title_pad = 22
    _tsne_axis_fs = 28
    _tsne_legend_fs = 18
    _tsne_tick_fs = 22
    _tsne_title_fp = fp.copy()
    _tsne_title_fp.set_size(_tsne_title_fs)
    _tsne_axis_fp = fp.copy()
    _tsne_axis_fp.set_size(_tsne_axis_fs)
    _tsne_leg_fp = fp.copy()
    _tsne_leg_fp.set_size(_tsne_legend_fs)
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    fig.patch.set_facecolor("white")

    for ax_idx, (task, sp_id) in enumerate(zip(TASK_HEADS, [0, 1])):
        ax = axes[ax_idx]
        ax.set_facecolor("white")
        mask = (species == sp_id)
        if not mask.any():
            continue
        class_names = TASK_META[task]["classes"]
        cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(class_names))
        for ci, cls in enumerate(class_names):
            cls_mask = mask & (labels == ci)
            if cls_mask.any():
                ax.scatter(emb_2d[cls_mask, 0], emb_2d[cls_mask, 1],
                           c=[cmap(ci)], label=cls, s=40, alpha=0.75, edgecolors="white", linewidths=0.3)
        if task == "dog_img":
            leg_loc, leg_anchor = "lower left", (0.02, 0.02)
        else:
            leg_loc, leg_anchor = "upper right", (0.98, 0.98)
        ncol = 2 if len(class_names) > 6 else 1
        ax.legend(
            prop=_tsne_leg_fp, loc=leg_loc, framealpha=0.9, markerscale=1.35, ncol=ncol,
            bbox_to_anchor=leg_anchor,
            handlelength=0.9, handletextpad=0.35, labelspacing=0.28, borderpad=0.32,
        )
        ax.set_xlabel("t-SNE 1", fontproperties=_tsne_axis_fp)
        ax.set_ylabel("t-SNE 2", fontproperties=_tsne_axis_fp)
        ax.set_title(task_titles_zh.get(task, task), fontproperties=_tsne_title_fp, pad=_tsne_title_pad)
        ax.tick_params(labelsize=_tsne_tick_fs)
        ax.margins(x=0.1, y=0.12)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.2)

    plt.tight_layout(pad=3.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"t-SNE 可视化已保存：{save_path}")

    del model
    torch.cuda.empty_cache()


# ============================================================
# GradCAM 可解释性分析
# ============================================================
def _get_gradcam_heatmap(model, image_tensor, target_class, target_layer, species_id: int = 0):
    """单张图像的 GradCAM 热力图，显式传入 species 确保路由正确"""
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    species_tensor = torch.tensor([species_id], device=image_tensor.device)
    output = model(image_tensor, species=species_tensor)
    task_key = "dog_img" if species_id == 0 else "cat_img"
    logits = output[task_key]
    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    handle_f.remove()
    handle_b.remove()

    act = activations["value"].squeeze(0)
    grad = gradients["value"].squeeze(0)

    if act.dim() == 3:
        weights = grad.mean(dim=(1, 2))
        cam = (weights[:, None, None] * act).sum(dim=0)
    else:
        cam = (grad * act).sum(dim=0)

    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()


def plot_gradcam_samples(model_state, dataset, test_indices, val_transform,
                         proc_mean, proc_std, device, save_path: Path,
                         n_samples_per_task: int = 4):
    model = DINOv3MultibranchModel().to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    if hasattr(model.backbone, 'stages'):
        stages = model.backbone.stages
    elif hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'stages'):
        stages = model.backbone.encoder.stages
    else:
        stages = None
    if stages is None:
        logger.warning("无法获取 backbone stages，跳过 GradCAM")
        del model
        return
    target_layer = stages[-1]

    inv_mean = [-m / s for m, s in zip(proc_mean, proc_std)]
    inv_std = [1.0 / s for s in proc_std]
    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=inv_std),
        transforms.Normalize(mean=inv_mean, std=[1, 1, 1]),
    ])

    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False
    task_zh = {"dog_img": "狗图像", "cat_img": "猫图像"}
    _gc_title_fs = 34
    _gc_title_pad = 22
    _gc_title_fp = fp.copy()
    _gc_title_fp.set_size(_gc_title_fs)

    fig, axes = plt.subplots(len(TASK_HEADS), n_samples_per_task * 2,
                              figsize=(7.5 * n_samples_per_task * 2, 7.5 * len(TASK_HEADS)))
    fig.patch.set_facecolor("white")
    if len(TASK_HEADS) == 1:
        axes = [axes]

    for task_row, task in enumerate(TASK_HEADS):
        sp_id = TASK_META[task]["species"]
        class_names = TASK_META[task]["classes"]

        task_indices = [i for i in test_indices
                        if dataset._species[i] == sp_id]
        if not task_indices:
            continue

        np.random.seed(SEED)
        chosen = np.random.choice(task_indices,
                                  min(n_samples_per_task, len(task_indices)),
                                  replace=False)

        for si, sample_idx in enumerate(chosen):
            path, label, sp, tn = dataset.samples[sample_idx]

            with Image.open(path) as img_pil:
                img_pil = img_pil.convert("RGB")
                img_tensor = val_transform(img_pil).unsqueeze(0).to(device)

            with torch.enable_grad():
                cam = _get_gradcam_heatmap(model, img_tensor, label, target_layer,
                                           species_id=sp)

            img_show = inv_normalize(img_tensor.squeeze(0).cpu())
            img_show = img_show.permute(1, 2, 0).numpy().clip(0, 1)

            import cv2
            cam_resized = cv2.resize(cam, (img_show.shape[1], img_show.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = heatmap[:, :, ::-1] / 255.0
            overlay = 0.5 * img_show + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)

            ax_orig = axes[task_row][si * 2]
            ax_cam = axes[task_row][si * 2 + 1]

            ax_orig.imshow(img_show)
            ax_orig.set_title(f"{task_zh.get(task, task)}: {class_names[label]}",
                              fontproperties=_gc_title_fp, pad=_gc_title_pad)
            ax_orig.axis("off")
            ax_orig.set_facecolor("white")

            ax_cam.imshow(overlay)
            ax_cam.set_title("Grad-CAM", fontproperties=_gc_title_fp, pad=_gc_title_pad)
            ax_cam.axis("off")
            ax_cam.set_facecolor("white")

        for si_extra in range(len(chosen), n_samples_per_task):
            axes[task_row][si_extra * 2].set_visible(False)
            axes[task_row][si_extra * 2 + 1].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"GradCAM 可视化已保存：{save_path}")

    del model
    torch.cuda.empty_cache()


# ============================================================
# 环境信息
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
# 单折训练 (v2: 渐进式解冻 + EMA)
# ============================================================
def train_fold(fold_idx, train_idx, val_idx, dataset,
               train_transform, val_transform, device, processor_mean, processor_std):
    seed_everything(SEED + fold_idx)

    # 训练集：猫类使用更强增强策略，狗类使用标准增强策略
    cat_train_transform = get_cat_train_transform(processor_mean, processor_std)
    train_dataset = CatAwareTransformSubset(dataset, train_idx, train_transform, cat_train_transform)
    val_dataset = TransformSubset(dataset, val_idx, val_transform)

    train_sampler = make_weighted_sampler(dataset, train_idx)
    n_workers = min(8, max(2, (os.cpu_count() or 4)))
    loader_kw = dict(
        batch_size=BATCH_SIZE,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
        prefetch_factor=(4 if n_workers > 0 else None),
    )

    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              drop_last=True, **loader_kw)

    # 不再传 class_weights：已有 WeightedRandomSampler 做 Batch 均衡，Loss 中叠加逆频率权重会过补偿
    species_criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
    task_criteria = {
        task: FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)
        for task in TASK_HEADS
    }

    val_loader_kw = dict(
        batch_size=BATCH_SIZE,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
        prefetch_factor=(4 if n_workers > 0 else None),
        shuffle=False,
    )
    val_loader = DataLoader(val_dataset, **val_loader_kw)

    model = DINOv3MultibranchModel().to(device)
    model.freeze_backbone()

    param_groups = model.get_param_groups(BACKBONE_LR, HEAD_LR, WEIGHT_DECAY,
                                          lr_decay=BACKBONE_LR_DECAY)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_warmup_cosine_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS,
                                            min_lr_ratio=0.005)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    steps_per_epoch = len(train_loader)
    ema_warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    ema = EMA(model, decay_start=EMA_DECAY_START, decay_end=EMA_DECAY_END,
              warmup_steps=ema_warmup_steps)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "lr": [], "val_balanced_f1": []}
    for t in TASK_HEADS:
        history[f"train_acc_{t}"] = []
        history[f"val_acc_{t}"] = []
        history[f"val_f1_{t}"] = []

    best_metric = 0.0
    patience_count = 0
    best_ema_state = None
    t_start = time.time()

    sorted_unfreeze = sorted(UNFREEZE_SCHEDULE.items())
    current_stage = "head_only"

    for epoch in range(1, NUM_EPOCHS + 1):
        t_ep = time.time()

        for ep_threshold, stage_name in sorted_unfreeze:
            if epoch == ep_threshold and stage_name != current_stage:
                if stage_name == "head_only":
                    model.freeze_backbone()
                else:
                    model.progressive_unfreeze(stage_name)
                current_stage = stage_name
                logger.info(f"  >>> Epoch {epoch}: 切换到 [{stage_name}]，"
                            f"解冻参数自动由 optimizer 接管（无需 add_param_group）")

        current_lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc, tr_task_acc, tr_task_f1 = run_epoch(
            model, train_loader, species_criterion, task_criteria,
            optimizer, device, scaler=scaler, epoch=epoch, ema=ema,
            use_augmix=True,
        )

        ema.apply_shadow(model)
        val_loss, val_acc, val_task_acc, val_task_f1 = run_epoch(
            model, val_loader, species_criterion, task_criteria,
            None, device,
        )
        ema.restore(model)
        scheduler.step()

        balanced_f1 = 0.5 * val_task_f1.get("dog_img", 0.0) + 0.5 * val_task_f1.get("cat_img", 0.0)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["val_balanced_f1"].append(balanced_f1)
        for t in TASK_HEADS:
            history[f"train_acc_{t}"].append(tr_task_acc.get(t, 0.0))
            history[f"val_acc_{t}"].append(val_task_acc.get(t, 0.0))
            history[f"val_f1_{t}"].append(val_task_f1.get(t, 0.0))

        elapsed = time.time() - t_ep
        task_str = " | ".join(
            f"{t}={val_task_acc.get(t,0)*100:.1f}%(F1={val_task_f1.get(t,0)*100:.1f}%)"
            for t in TASK_HEADS
        )
        logger.info(
            f"[折{fold_idx+1}] Epoch [{epoch:>3d}/{NUM_EPOCHS}] "
            f"训练: loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"验证(EMA): loss={val_loss:.4f} acc={val_acc*100:.2f}% balanced_F1={balanced_f1*100:.2f}% | "
            f"{task_str} | lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        if balanced_f1 > best_metric + 1e-4:
            best_metric = balanced_f1
            patience_count = 0
            best_ema_state = copy.deepcopy(ema.state_dict())
            logger.info(f"  >>> 最优 EMA 权重已保存 (balanced_F1={balanced_f1*100:.2f}%)")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"\n早停触发：balanced_F1 连续 {PATIENCE} 轮未改善")
                break

    elapsed_total = time.time() - t_start
    logger.info(f"[折{fold_idx+1}] 完成: {len(history['train_loss'])} 轮, "
                f"最优 balanced_F1(EMA)={best_metric*100:.2f}%, 用时 {elapsed_total/60:.1f} 分钟")

    del model
    torch.cuda.empty_cache()
    return best_ema_state, history


# ============================================================
# 主流程
# ============================================================
def main():
    setup_run_logging()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # PyTorch 2.x：Tensor Core 上矩阵乘用 TF32，通常加速且对微调影响很小
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    logger.info("=" * 60)
    logger.info("大创项目 DINOv3 ConvNeXt-Tiny 端到端微调（图像情绪识别）")
    logger.info("=" * 60)
    log_environment()
    logger.info(f"计算设备：{device}")
    logger.info(f"模型：{DINOV3_MODEL_NAME}")
    logger.info(f"超参数：epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, "
                f"backbone_lr={BACKBONE_LR}, backbone_lr_decay={BACKBONE_LR_DECAY}, "
                f"head_lr={HEAD_LR}, wd={WEIGHT_DECAY}, "
                f"patience={PATIENCE}, ema_decay={EMA_DECAY_START}→{EMA_DECAY_END}, "
                f"mixup_alpha={MIXUP_ALPHA}, "
                f"focal_gamma={FOCAL_GAMMA}, cat_task_weight={CAT_TASK_WEIGHT}, "
                f"n_folds={N_FOLDS}, tta_steps={TTA_STEPS}")
    logger.info(f"早停指标：balanced_F1 = 0.5×dog_macro_f1 + 0.5×cat_macro_f1")
    logger.info(f"渐进解冻策略：{UNFREEZE_SCHEDULE}")
    logger.info("=" * 60)

    # ── 1. 获取 DINOv3 图像预处理参数 ────────────────────────
    logger.info("\n[1/8] 加载 DINOv3 图像预处理器...")
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_NAME)
    proc_mean = processor.image_mean
    proc_std  = processor.image_std
    logger.info(f"  预处理参数: mean={proc_mean}, std={proc_std}")

    train_transform = get_train_transform(proc_mean, proc_std)
    val_transform   = get_val_transform(proc_mean, proc_std)
    tta_transforms  = get_tta_transforms(proc_mean, proc_std)

    # ── 2. 加载数据集 ─────────────────────────────────────────
    logger.info("\n[2/8] 加载图像数据集...")
    DOG_IMG_DIR = ROOT / "data" / "dog_emotion_cropped"
    CAT_IMG_DIR = ROOT / "data" / "cat_671_cropped"

    data_dirs = {
        "dog_img": (DOG_IMG_DIR, DOG_IMG_CLASSES),
        "cat_img": (CAT_IMG_DIR, CAT_IMG_CLASSES),
    }
    dataset = PetImageDataset(data_dirs, transform=val_transform)

    # ── 3. 数据划分 ──────────────────────────────────────────
    logger.info("\n[3/8] 分层采样划分数据集...")
    train_idx, val_idx, test_idx = stratified_split(dataset, TRAIN_RATIO, VAL_RATIO, SEED)
    trainval_idx = np.concatenate([train_idx, val_idx])
    logger.info(f"  训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")
    logger.info(f"  训练模式：{'%d 折交叉验证集成' % N_FOLDS if N_FOLDS > 1 else '单次训练'}")

    # ── 4. 训练 ───────────────────────────────────────────────
    logger.info("\n[4/8] 开始训练...")
    fold_states = []
    fold_histories = []

    if N_FOLDS > 1:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        tv_labels = dataset.stratify_labels[trainval_idx]
        for fold_idx, (tr_sub, vl_sub) in enumerate(skf.split(trainval_idx, tv_labels)):
            fold_train = trainval_idx[tr_sub]
            fold_val   = trainval_idx[vl_sub]
            logger.info(f"\n{'─'*50}")
            logger.info(f"第 {fold_idx+1}/{N_FOLDS} 折  训练={len(fold_train)}, 验证={len(fold_val)}")
            logger.info(f"{'─'*50}")
            state, hist = train_fold(
                fold_idx, fold_train, fold_val, dataset,
                train_transform, val_transform, device, proc_mean, proc_std,
            )
            fold_states.append(state)
            fold_histories.append(hist)
    else:
        state, hist = train_fold(
            0, train_idx, val_idx, dataset,
            train_transform, val_transform, device, proc_mean, proc_std,
        )
        fold_states.append(state)
        fold_histories.append(hist)

    # ── 5. 绘制训练曲线 ──────────────────────────────────────
    logger.info("\n[5/8] 绘制训练曲线...")
    fig_path = FIG_DIR / f"dinov3_train_curve_{TS}.png"
    plot_training_curves(fold_histories[-1], fig_path)

    # ── 6. TTA 评估 ────────────────────────────────────────
    n_tta = len(tta_transforms)
    logger.info(f"\n[6/8] TTA 评估（{N_FOLDS} 折 × {n_tta} TTA 视角）...")
    cm_data, eval_probs, eval_targets = ensemble_evaluation(
        fold_states, dataset, test_idx, tta_transforms, device,
    )
    if cm_data:
        cm_path = FIG_DIR / f"dinov3_confusion_matrix_{TS}.png"
        plot_confusion_matrices(cm_data, cm_path)

    # ROC 曲线
    logger.info("  绘制 ROC 曲线...")
    try:
        roc_path = FIG_DIR / f"dinov3_roc_{TS}.png"
        class_names_map = {t: dataset.task_class_names[t] for t in TASK_HEADS}
        plot_roc_curves(eval_probs, eval_targets, class_names_map, roc_path)
    except Exception as e:
        logger.warning(f"ROC 曲线绘制失败：{e}")

    # ── 保存模型 ─────────────────────────────────────────────
    model_path = MODEL_DIR / f"DINOv3_ConvNeXt_{TS}.pkl"
    torch.save({
        "model_states": fold_states,
        "n_folds": N_FOLDS,
        "model_name": DINOV3_MODEL_NAME,
        "hyperparams": {
            "backbone_lr": BACKBONE_LR, "backbone_lr_decay": BACKBONE_LR_DECAY,
            "head_lr": HEAD_LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
            "unfreeze_schedule": UNFREEZE_SCHEDULE,
            "mixup_alpha": MIXUP_ALPHA,
            "focal_gamma": FOCAL_GAMMA, "label_smooth": LABEL_SMOOTH,
            "cat_task_weight": CAT_TASK_WEIGHT,
            "ema_decay_start": EMA_DECAY_START, "ema_decay_end": EMA_DECAY_END,
            "n_folds": N_FOLDS, "tta_steps": TTA_STEPS,
            "seed": SEED,
        },
        "task_meta": TASK_META,
    }, model_path)
    logger.info(f"模型已保存（含 {N_FOLDS} 折权重）：{model_path}")

    # ── 7. t-SNE 特征可视化 ──────────────────────────────────
    logger.info("\n[7/8] t-SNE 特征可视化...")
    try:
        tsne_path = FIG_DIR / f"dinov3_tsne_{TS}.png"
        plot_tsne_features(fold_states[0], dataset, test_idx, val_transform,
                           device, tsne_path)
    except Exception as e:
        logger.warning(f"t-SNE 可视化失败：{e}")

    # ── 8. GradCAM 可解释性分析 ──────────────────────────────
    logger.info("\n[8/8] GradCAM 可解释性分析...")
    try:
        gradcam_path = FIG_DIR / f"dinov3_gradcam_{TS}.png"
        plot_gradcam_samples(fold_states[0], dataset, test_idx, val_transform,
                             proc_mean, proc_std, device, gradcam_path)
    except Exception as e:
        logger.warning(f"GradCAM 可视化失败：{e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"训练日志：{log_path}")
    logger.info(f"模型文件：{model_path}")
    logger.info(f"ROC 曲线：{FIG_DIR / f'dinov3_roc_{TS}.png'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Windows 多进程 DataLoader 必须在 main guard 内启动
    import multiprocessing
    multiprocessing.freeze_support()
    main()
