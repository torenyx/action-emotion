# -*- coding: utf-8 -*-
"""
模型定义：CED-Mini 骨干 + 多任务分类头。

架构设计：
  ┌─────────────────────────────────────────────┐
  │  CED-Mini Encoder (冻结/微调, embed_dim=256) │
  │  12 层 ViT Transformer, 4 heads             │
  │  输入: (B, n_mels=64, T)                     │
  │  输出: (B, N_patches, 256)  → mean pool      │
  │        → (B, 256)                            │
  └───────────────┬─────────────────────────────┘
                  │
          ┌───────┴───────┐
          │  共享投影层    │
          │  256 → 512    │
          │  LayerNorm    │
          │  GELU         │
          │  ResBlock×2   │
          └───────┬───────┘
          ┌───────┴───────┐
    ┌─────┴────┐   ┌─────┴────┐   ┌─────────────┐
    │ Species  │   │ Dog Head │   │ Cat Head    │
    │ 512→2    │   │ 512→4    │   │ 512→10      │
    └──────────┘   └──────────┘   └─────────────┘

微调策略：
  - 默认冻结 CED-Mini 底层 8 层，只解冻顶层 4 层 + LayerNorm + BN
  - 骨干参数使用小学习率 (backbone_lr)，分类头使用大学习率 (head_lr)
  - 支持完全冻结 / 完全解冻 / 渐进解冻
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification

from .config import TrainConfig, TASK_META, CED_MODEL_DIR

logger = logging.getLogger("cedtrain")


# ============================================================
# 残差 MLP 块
# ============================================================

class ResidualBlock(nn.Module):
    """LayerNorm + GELU 残差块，适用于小 batch。"""

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


# ============================================================
# 任务分类头
# ============================================================

class TaskHead(nn.Module):
    """两层分类头，为多类别任务提供足够建模能力。"""

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


# ============================================================
# CED-Mini 音频情绪分类模型
# ============================================================

class CedAudioEmotionModel(nn.Module):
    """
    CED-Mini 骨干 + 共享投影 + 多任务分类头。

    输入: (B, n_mels, T) log-mel 频谱
    输出: dict["species"]→(B,2)；「狗/猫情绪」头与 train/3 一致：
      训练传入 species=task_id 时用 GT 硬路由；推理 species=None 时用预测物种路由。
    """

    HIDDEN_DIM = 512
    FREEZE_BOTTOM_LAYERS = 8

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        ced_model = AutoModelForAudioClassification.from_pretrained(
            str(CED_MODEL_DIR), trust_remote_code=True, local_files_only=True,
        )
        self.encoder = ced_model.encoder
        self.embed_dim = ced_model.config.embed_dim  # 256

        del ced_model
        logger.info(f"CED-Mini 编码器加载完毕, embed_dim={self.embed_dim}")

        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.HIDDEN_DIM),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(self.HIDDEN_DIM, dropout=0.2),
            ResidualBlock(self.HIDDEN_DIM, dropout=0.2),
        )

        # 物种二分类：0=狗音频任务，1=猫音频任务（与 task_id 一致）
        self.head_species = nn.Linear(self.HIDDEN_DIM, 2)

        self.heads = nn.ModuleDict()
        for task_name, meta in TASK_META.items():
            drop = cfg.task_dropouts.get(task_name, 0.3)
            self.heads[task_name] = TaskHead(
                self.HIDDEN_DIM, meta["num_classes"], dropout=drop,
            )

        self._setup_freeze()

    # 骨干共 12 层 block（0-indexed）
    TOTAL_BLOCKS = 12

    def _setup_freeze(self) -> None:
        """初始化时全部冻结骨干，渐进解冻由 set_unfreeze_stage() 动态控制。"""
        for param in self.encoder.parameters():
            param.requires_grad = False

        # LN / BN 始终保持可训（适配下游数据分布）
        for name, param in self.encoder.named_parameters():
            if "init_bn" in name:
                param.requires_grad = True
            elif "norm" in name and "blocks" not in name:
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            f"CED-Mini 骨干初始化: 骨干全冻 (仅 LN/BN 可训), "
            f"可训参数 {trainable:,}/{total:,} ({trainable / total * 100:.1f}%)"
        )

    def set_unfreeze_stage(self, stage: str) -> None:
        """
        按渐进解冻策略动态调整骨干可训参数。

        stage 取值：
          "head_only" → 骨干 block 全冻（仅 LN/BN 可训）
          "top2"      → 解冻顶层 2 个 block (block 10-11) + LN
          "top4"      → 解冻顶层 4 个 block (block 8-11) + LN（= 原静态策略）
        """
        freeze_below = {
            "head_only": self.TOTAL_BLOCKS,   # 全冻
            "top2":      self.TOTAL_BLOCKS - 2,  # 解冻 block 10-11
            "top4":      self.TOTAL_BLOCKS - 4,  # 解冻 block 8-11
        }
        threshold = freeze_below.get(stage, self.FREEZE_BOTTOM_LAYERS)

        for name, param in self.encoder.named_parameters():
            if "init_bn" in name or ("norm" in name and "blocks" not in name):
                param.requires_grad = True
                continue
            if "blocks." in name:
                block_idx = int(name.split("blocks.")[1].split(".")[0])
                param.requires_grad = block_idx >= threshold
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            f"  [解冻] 策略={stage}, 骨干可训 {trainable:,}/{total:,} "
            f"({trainable / total * 100:.1f}%)"
        )

    def get_param_groups(self, cfg: TrainConfig) -> list[dict]:
        """
        差异化学习率参数组，骨干内部按层级衰减：
          顶层 block LR = backbone_lr
          每往下一层 × backbone_lr_decay
          LN/BN 等非 block 参数使用顶层 LR
        """
        # 按 block 索引收集骨干可训参数
        block_params: dict[int, list] = {}
        other_backbone_params: list = []

        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if "blocks." in name:
                block_idx = int(name.split("blocks.")[1].split(".")[0])
                block_params.setdefault(block_idx, []).append(param)
            else:
                other_backbone_params.append(param)

        groups: list[dict] = []
        top_block = self.TOTAL_BLOCKS - 1  # 最高层 block 索引（= 11）

        for block_idx in sorted(block_params.keys(), reverse=True):
            depth = top_block - block_idx                        # 距顶层的层数
            lr_i = cfg.backbone_lr * (cfg.backbone_lr_decay ** depth)
            groups.append({
                "params": block_params[block_idx],
                "lr": lr_i,
                "weight_decay": cfg.weight_decay,
            })

        if other_backbone_params:
            groups.append({
                "params": other_backbone_params,
                "lr": cfg.backbone_lr,
                "weight_decay": cfg.weight_decay,
            })

        head_params = (
            list(self.projection.parameters())
            + list(self.res_blocks.parameters())
            + list(self.head_species.parameters())
            + list(self.heads.parameters())
        )
        groups.append({
            "params": head_params,
            "lr": cfg.head_lr,
            "weight_decay": cfg.weight_decay,
        })
        return groups

    def forward(
        self,
        mel: torch.Tensor,
        species: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            mel: (B, n_mels, T) log-mel 频谱
            species: (B,) 物种标签 0=狗 1=猫，与 batch['task_id'] 一致；训练/验证传入以 GT 路由。
                     推理传 None，用 head_species 预测做硬路由。

        Returns:
            dict: species → (B, 2)；dog_audio / cat_audio → (B, num_classes)，非路由行 logits 为 0。
        """
        encoder_out = self.encoder(mel)
        # CED 官方 CedModel.forward 返回 SequenceClassifierOutput，字段名为 logits，
        # 实为 token 序列隐状态 (B, N, D)，非分类 logits；见 modeling_ced.py L447、L535。
        token_hidden = encoder_out.logits
        if token_hidden.dim() != 3:
            raise RuntimeError(
                f"CED 编码器输出应为 (B, N, D) 序列隐状态，当前 shape={tuple(token_hidden.shape)}，"
                "若上游改为直接输出分类 logits，请同步修改 pooling 与下游头。",
            )
        if token_hidden.size(-1) != self.embed_dim:
            raise RuntimeError(
                f"隐状态最后一维应为 embed_dim={self.embed_dim}，当前 D={token_hidden.size(-1)}。",
            )
        pooled = token_hidden.mean(dim=1)  # (B, embed_dim) mean pooling

        h = self.projection(pooled)
        h = self.res_blocks(h)

        species_logits = self.head_species(h)
        if species is not None:
            route = species.long()
        else:
            route = species_logits.argmax(dim=1)

        bsz = h.shape[0]
        n_dog = TASK_META["dog_audio"]["num_classes"]
        n_cat = TASK_META["cat_audio"]["num_classes"]
        dog_logits = h.new_zeros(bsz, n_dog)
        cat_logits = h.new_zeros(bsz, n_cat)
        dog_mask = route == 0
        cat_mask = route == 1
        if dog_mask.any():
            dog_logits[dog_mask] = self.heads["dog_audio"](h[dog_mask]).to(dog_logits.dtype)
        if cat_mask.any():
            cat_logits[cat_mask] = self.heads["cat_audio"](h[cat_mask]).to(cat_logits.dtype)

        return {
            "species": species_logits,
            "dog_audio": dog_logits,
            "cat_audio": cat_logits,
        }

    @torch.no_grad()
    def encode_shared_features(self, mel: torch.Tensor) -> torch.Tensor:
        """
        共享投影 + ResBlock 之后的 512 维表征，用于 t-SNE 等可解释性可视化。
        与情绪头路由无关，对 batch 内全部样本计算。
        """
        encoder_out = self.encoder(mel)
        token_hidden = encoder_out.logits
        if token_hidden.dim() != 3:
            raise RuntimeError(
                f"CED 编码器输出应为 (B, N, D)，当前 shape={tuple(token_hidden.shape)}",
            )
        pooled = token_hidden.mean(dim=1)
        h = self.projection(pooled)
        h = self.res_blocks(h)
        return h
