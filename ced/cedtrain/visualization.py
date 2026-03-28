# -*- coding: utf-8 -*-
"""
可视化：训练曲线、混淆矩阵热力图。

遵循项目规范：
  - 白色背景、无网格线
  - 中文字体 ≥20 号
  - 图例/坐标轴/标签无遮挡
  - 保存到 figure/ 目录
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from .config import TASK_META

TASK_NAMES = list(TASK_META.keys())
TASK_ZH = {"dog_audio": "狗音频情绪", "cat_audio": "猫音频情绪"}

# 绘图统一字号（报告可读性）
_LEGEND_FS = 22
_TICK_FS = 20
_AXIS_LABEL_FS = 20
# 第二行各任务子图：图例与曲线错开（狗：右下易挡线尾 → 左上；猫：左上易挡上升段 → 右下）
_TASK_LEGEND_LOC = {"dog_audio": "upper left", "cat_audio": "lower right"}

# 混淆矩阵：标题适中，轴名与刻度清晰
_CM_TITLE_FS = 24
_CM_AXIS_NAME_FS = 22
_CM_CLASS_TICK_FS = 16


def _get_zh_font() -> fm.FontProperties:
    zh_fonts = [
        f.fname for f in fm.fontManager.ttflist
        if any(kw in f.name for kw in
               ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong"])
    ]
    if zh_fonts:
        return fm.FontProperties(fname=zh_fonts[0])
    return fm.FontProperties()


def plot_training_curves(history: dict, save_path: Path) -> None:
    """绘制训练曲线：总体 Loss / 平衡宏 F1 / LR + 各任务 macro F1（早停与最优权重同指标）。"""
    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    epochs = range(1, len(history["train_loss"]) + 1)
    n_tasks = len(TASK_NAMES)
    fig, axes = plt.subplots(2, max(n_tasks, 3), figsize=(8 * max(n_tasks, 3), 13))
    fig.patch.set_facecolor("white")

    line_kw = dict(linewidth=2.2, marker="o", markersize=3)
    colors = {"train": "#2E86AB", "val": "#E84855"}

    def style_ax(ax):
        ax.set_facecolor("white")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.2)

    # 第一行：总体 Loss / 平衡宏 F1 / 学习率
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color=colors["train"], label="训练", **line_kw)
    ax.plot(epochs, history["val_loss"], color=colors["val"], label="验证", **line_kw)
    ax.set_xlabel("Epoch", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
    ax.set_ylabel("Loss", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
    ax.legend(fontsize=_LEGEND_FS, prop=fp, framealpha=0.92)
    ax.tick_params(labelsize=_TICK_FS)
    style_ax(ax)

    ax = axes[0, 1]
    tr_bf = history.get("train_balanced_macro_f1", history.get("train_acc", []))
    vl_bf = history.get("val_balanced_macro_f1", history.get("val_acc", []))
    ax.plot(epochs, [a * 100 for a in tr_bf], color=colors["train"], label="训练", **line_kw)
    ax.plot(epochs, [a * 100 for a in vl_bf], color=colors["val"], label="验证", **line_kw)
    ax.set_xlabel("Epoch", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
    ax.set_ylabel("平衡宏 F1 (%)", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
    ax.legend(fontsize=_LEGEND_FS, prop=fp, framealpha=0.92)
    ax.tick_params(labelsize=_TICK_FS)
    style_ax(ax)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, max(y1 * 1.08, y1 + 5))

    ax = axes[0, 2]
    ax.plot(epochs, history["lr"], color="#6C5B7B", linewidth=2.5)
    ax.set_xlabel("Epoch", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
    ax.set_ylabel("学习率", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
    ax.tick_params(labelsize=_TICK_FS)
    style_ax(ax)

    for col in range(3, axes.shape[1]):
        axes[0, col].set_visible(False)

    # 第二行：各任务宏 F1
    for idx, task_name in enumerate(TASK_NAMES):
        ax = axes[1, idx]
        train_key = f"train_macro_f1_{task_name}"
        val_key = f"val_macro_f1_{task_name}"
        if train_key in history and val_key in history:
            ax.plot(epochs, [a * 100 for a in history[train_key]],
                    color=colors["train"], label="训练", **line_kw)
            ax.plot(epochs, [a * 100 for a in history[val_key]],
                    color=colors["val"], label="验证", **line_kw)
        ax.set_xlabel("Epoch", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
        ax.set_ylabel("宏 F1 (%)", fontsize=_AXIS_LABEL_FS, fontproperties=fp)
        ax.set_title(
            TASK_ZH.get(task_name, task_name),
            fontsize=24, fontproperties=fp, pad=18, loc="left",
        )
        ax.legend(
            fontsize=_LEGEND_FS, prop=fp, framealpha=0.92,
            loc=_TASK_LEGEND_LOC.get(task_name, "best"),
        )
        ax.tick_params(labelsize=_TICK_FS)
        ax.set_ylim(0, 118)
        style_ax(ax)

    for col in range(len(TASK_NAMES), axes.shape[1]):
        axes[1, col].set_visible(False)

    plt.tight_layout(pad=2.5)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_all_fold_val_curves(fold_histories: list[dict], save_path: Path) -> None:
    """绘制各折验证集 balanced_macro_F1，用于观察折间方差。"""
    if not fold_histories:
        return
    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_linewidth(1.2)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(fold_histories)))
    for i, hist in enumerate(fold_histories):
        vl = hist.get("val_balanced_macro_f1", [])
        if not vl:
            continue
        ep = range(1, len(vl) + 1)
        ax.plot(
            ep, [a * 100 for a in vl],
            linewidth=2.2, marker="o", markersize=3,
            color=colors[i], label=f"第 {i + 1} 折（验证）",
        )

    ax.set_xlabel("Epoch", fontsize=20, fontproperties=fp)
    ax.set_ylabel("平衡宏 F1 (%)", fontsize=20, fontproperties=fp)
    ax.legend(fontsize=_LEGEND_FS, prop=fp, framealpha=0.92, loc="best")
    ax.tick_params(labelsize=_TICK_FS)
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_confusion_matrices(cm_data: dict, save_path: Path) -> None:
    """绘制各任务混淆矩阵热力图。"""
    fp = _get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    n_tasks = len(cm_data)
    if n_tasks == 0:
        return

    # 猫音频有 10 类，需要更大的子图空间避免单元格文字超出
    task_widths = []
    for task_name in cm_data:
        n_cls = len(cm_data[task_name][1])
        task_widths.append(max(7, n_cls * 0.85))
    total_w = sum(task_widths) + (n_tasks - 1) * 0.5
    fig, axes = plt.subplots(1, n_tasks, figsize=(total_w, max(task_widths) * 0.9 + 1.5))
    fig.patch.set_facecolor("white")
    if n_tasks == 1:
        axes = [axes]

    for ax, (task_name, (cm, class_names)) in zip(axes, cm_data.items()):
        cm_pct = cm.astype(np.float64)
        row_sums = np.maximum(cm_pct.sum(axis=1, keepdims=True), 1)
        cm_pct = cm_pct / row_sums * 100

        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        n = len(class_names)
        # 类别越多字越小，猫音频(10类)用更小字号
        cell_fs = max(8, 16 - n)
        for i in range(n):
            for j in range(n):
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(
                    j, i, f"{cm_pct[i, j]:.1f}%\n({cm[i, j]})",
                    ha="center", va="center",
                    fontsize=cell_fs,
                    color=color, fontproperties=fp,
                )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(
            class_names, fontsize=_CM_CLASS_TICK_FS, fontproperties=fp, rotation=45, ha="right",
        )
        ax.set_yticklabels(class_names, fontsize=_CM_CLASS_TICK_FS, fontproperties=fp)
        ax.set_xlabel("预测", fontsize=_CM_AXIS_NAME_FS, fontproperties=fp)
        ax.set_ylabel("真实", fontsize=_CM_AXIS_NAME_FS, fontproperties=fp)
        title_fp = fp.copy()
        title_fp.set_size(_CM_TITLE_FS)
        ax.set_title(TASK_ZH.get(task_name, task_name), fontproperties=title_fp, pad=18)
        ax.set_facecolor("white")
        for s in ax.spines.values():
            s.set_linewidth(1.2)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
