# -*- coding: utf-8 -*-
"""
CED 音频单模态：与 ImageBind/LanguageBind 对齐的可解释性组图。

  - 各类别精确率 / 召回率 / F1 分组柱状图（狗音频 / 猫音频 双子图）
  - 多分类微平均 ROC 曲线与 AUC
  - 预测置信度分布（正确 vs 错误）
  - 混淆矩阵非对角线 Top-K 错误对条形图
  - 共享 512 维表征的 t-SNE（测试集，按物种分面着色）
  - Mel 频谱 GradCAM（Token-wise Grad-CAM，对 ViT token 梯度 reshape 回 Mel 网格）

无网格线、白底、中文标签，图保存至 figure/，文件名含时间戳。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

from .config import TASK_META, TrainConfig
from .data import AudioEmotionDataset, make_collate_fn
from .engine import batch_to_device
from .modeling import CedAudioEmotionModel
from .utils import get_zh_font

logger = logging.getLogger("cedtrain")

TASK_HEADS: tuple[str, ...] = tuple(TASK_META.keys())

TASK_ZH = {
    "dog_audio": "狗音频情绪",
    "cat_audio": "猫音频情绪",
}

# 可解释性组图：统一偏大字号，避免报告看不清
_LEGEND_FS = 26
_TICK_FS = 20
_CORNER_TITLE_FS = 30   # 左上角任务标题字号，需在报告中清晰可见
_BAR_YTICK_FS = 18


def _title_font_prop(fp, size: int):
    """标题用 FontProperties：须 set_size，否则与 fontsize= 同时传入时可能被默认字号覆盖。"""
    t = fp.copy()
    t.set_size(size)
    return t


def _legend_prop(fp):
    p = fp.copy()
    p.set_size(_LEGEND_FS)
    return p


def _style_ax(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_linewidth(1.2)


def plot_per_class_prf_bars(
    prob_bundle: dict[str, tuple[np.ndarray, np.ndarray]],
    task_class_names: dict[str, list[str]],
    save_path: Path,
) -> None:
    """各类别 P/R/F1 分组柱状图，1×2 子图（狗 / 猫）。"""
    fp = get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    fig.patch.set_facecolor("white")
    axes = np.ravel(axes)

    for ax, task in zip(axes, TASK_HEADS):
        if task not in prob_bundle:
            ax.set_visible(False)
            continue
        probs, targets = prob_bundle[task]
        preds = probs.argmax(axis=1)
        class_names = task_class_names[task]
        n_cls = len(class_names)
        labels = list(range(n_cls))
        p, r, f1, _ = precision_recall_fscore_support(
            targets, preds, average=None, labels=labels, zero_division=0,
        )

        x = np.arange(n_cls)
        w = 0.24
        ax.bar(x - w, p * 100, w, label="精确率", color="#2E86AB", alpha=0.88)
        ax.bar(x, r * 100, w, label="召回率", color="#E84855", alpha=0.88)
        ax.bar(x + w, f1 * 100, w, label="F1", color="#6C5B7B", alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontproperties=fp, fontsize=18, rotation=35, ha="right")
        ax.set_ylabel("百分比 (%)", fontproperties=fp, fontsize=24)
        # 数据范围 0–100%；狗/猫图例均右上，避免与左侧标题重叠；图例略放大便于阅读
        leg_kw = dict(
            prop=_legend_prop(fp),
            framealpha=0.92,
            ncol=3,
            columnspacing=0.88,
            handlelength=2.1,
            handleheight=1.05,
            handletextpad=0.52,
            borderpad=0.45,
            labelspacing=0.45,
            markerscale=1.2,
        )
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), **leg_kw)
        ax.tick_params(labelsize=_TICK_FS)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        _style_ax(ax)
        ax.set_title(
            TASK_ZH.get(task, task),
            loc="left", fontproperties=_title_font_prop(fp, _CORNER_TITLE_FS), pad=28,
        )

    plt.tight_layout(pad=2.8)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("已保存各类别 P/R/F1 柱状图：%s", save_path)


def plot_micro_roc_curves(
    prob_bundle: dict[str, tuple[np.ndarray, np.ndarray]],
    task_class_names: dict[str, list[str]],
    save_path: Path,
) -> None:
    """各任务多分类微平均 ROC（OvR 展平）与 AUC。"""
    fp = get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    fig.patch.set_facecolor("white")
    axes = np.ravel(axes)
    colors = ["#2E86AB", "#E84855"]

    for ax, task, color in zip(axes, TASK_HEADS, colors):
        if task not in prob_bundle:
            ax.set_visible(False)
            continue
        probs, targets = prob_bundle[task]
        class_names = task_class_names[task]
        n_cls = len(class_names)
        if n_cls < 2:
            ax.text(0.5, 0.5, "类别数不足", ha="center", va="center", fontproperties=fp, fontsize=22)
            _style_ax(ax)
            continue

        y_bin = label_binarize(targets, classes=np.arange(n_cls))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=3.0, label=f"微平均 AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.45)
        ax.set_xlabel("假正率", fontproperties=fp, fontsize=24)
        ax.set_ylabel("真正率", fontproperties=fp, fontsize=24)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.12)
        ax.legend(prop=_legend_prop(fp), framealpha=0.92, loc="lower right", markerscale=1.15)
        ax.tick_params(labelsize=_TICK_FS)
        _style_ax(ax)
        ax.set_title(
            TASK_ZH.get(task, task),
            loc="left", fontproperties=_title_font_prop(fp, _CORNER_TITLE_FS), pad=22,
        )

    plt.tight_layout(pad=2.8)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("已保存微平均 ROC 曲线：%s", save_path)


def plot_confidence_distributions(
    prob_bundle: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """各任务预测最大概率分布：正确 vs 错误；图例横排置于子图轴框上方。"""
    fp = get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor("white")
    axes = np.ravel(axes)

    for ax, task in zip(axes, TASK_HEADS):
        if task not in prob_bundle:
            ax.set_visible(False)
            continue
        probs, targets = prob_bundle[task]
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        ok = preds == targets
        ax.hist(
            conf[ok], bins=36, alpha=0.72, color="#2E86AB", label="预测正确", density=True,
        )
        ax.hist(
            conf[~ok], bins=36, alpha=0.72, color="#E84855", label="预测错误", density=True,
        )
        ax.set_xlabel("最大置信度", fontproperties=fp, fontsize=24)
        ax.set_ylabel("密度", fontproperties=fp, fontsize=24)
        y_hi = ax.get_ylim()[1]
        ax.set_ylim(0, y_hi * 1.06)
        leg_kw = dict(
            prop=_legend_prop(fp),
            framealpha=0.92,
            ncol=2,
            columnspacing=1.0,
            handlelength=2.1,
            handleheight=1.05,
            handletextpad=0.52,
            borderpad=0.45,
            labelspacing=0.45,
            markerscale=1.2,
        )
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), **leg_kw)
        ax.tick_params(labelsize=_TICK_FS)
        _style_ax(ax)
        ax.set_title(
            TASK_ZH.get(task, task),
            loc="left", fontproperties=_title_font_prop(fp, _CORNER_TITLE_FS), pad=22,
        )

    plt.tight_layout(pad=2.8)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("已保存置信度分布图：%s", save_path)


def plot_top_confusion_pairs(
    cm_data: dict[str, tuple[np.ndarray, list[str]]],
    save_path: Path,
    top_k: int = 12,
) -> None:
    """各任务混淆矩阵中非对角元素 Top-K（按错误计数）。"""
    fp = get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    fig.patch.set_facecolor("white")
    axes = np.ravel(axes)

    for ax, task in zip(axes, TASK_HEADS):
        if task not in cm_data:
            ax.set_visible(False)
            continue
        cm, class_names = cm_data[task]
        n = len(class_names)
        pairs: list[tuple[int, str, int]] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c = int(cm[i, j])
                if c > 0:
                    pairs.append((c, f"{class_names[i]}→{class_names[j]}", i * n + j))
        pairs.sort(key=lambda x: -x[0])
        pairs = pairs[:top_k]
        if not pairs:
            ax.axis("off")
            ax.text(
                0.5, 0.88, TASK_ZH.get(task, task),
                transform=ax.transAxes, fontproperties=_title_font_prop(fp, _CORNER_TITLE_FS),
                ha="center", va="top",
            )
            ax.text(
                0.5, 0.48,
                "无非对角错误\n（该任务测试集无此类混淆）",
                ha="center", va="center", fontproperties=fp, fontsize=24,
            )
            continue
        counts = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        y = np.arange(len(counts))
        ax.barh(y, counts, color="#6C5B7B", alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontproperties=fp, fontsize=_BAR_YTICK_FS)
        ax.invert_yaxis()
        ax.set_xlabel("错误样本数", fontproperties=fp, fontsize=24)
        ax.margins(y=0.04)
        ax.tick_params(labelsize=_TICK_FS)
        _style_ax(ax)
        ax.set_title(
            TASK_ZH.get(task, task),
            loc="left", fontproperties=_title_font_prop(fp, _CORNER_TITLE_FS), pad=22,
        )

    plt.tight_layout(pad=2.8)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("已保存 Top 混淆对条形图：%s", save_path)


def run_mlp_interpretability_suite(
    cm_data: dict[str, tuple[np.ndarray, list[str]]],
    prob_bundle: dict[str, tuple[np.ndarray, np.ndarray]],
    task_class_names: dict[str, list[str]],
    fig_dir: Path,
    ts: str,
    prefix: str = "ced_",
) -> None:
    """
    在集成评估得到 cm_data 与 prob_bundle 后，一次性生成四类可解释性图。

    prefix 默认 \"ced_\"，与 figure/ced_interpret_*_{ts}.png 对应。
    """
    if not prob_bundle:
        logger.warning("prob_bundle 为空，跳过可解释性组图")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    p = prefix

    plot_per_class_prf_bars(
        prob_bundle, task_class_names, fig_dir / f"{p}interpret_prf_{ts}.png",
    )
    plot_micro_roc_curves(
        prob_bundle, task_class_names, fig_dir / f"{p}interpret_roc_{ts}.png",
    )
    plot_confidence_distributions(
        prob_bundle, fig_dir / f"{p}interpret_confidence_{ts}.png",
    )
    if cm_data:
        plot_top_confusion_pairs(
            cm_data, fig_dir / f"{p}interpret_top_confusion_{ts}.png",
        )


@torch.no_grad()
def plot_tsne_ced_audio(
    model_state: dict,
    dataset: AudioEmotionDataset,
    test_indices: np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
    save_path: Path,
    max_samples: int = 800,
    seed: int = 42,
) -> None:
    """
    测试集共享 512 维表征的 t-SNE；左：狗音频按类着色，右：猫音频按类着色。
    使用单次 t-SNE 嵌入，与子图掩码一致（与 DINOv3 图像脚本思路对齐）。
    """
    rng = np.random.default_rng(seed)
    model = CedAudioEmotionModel(cfg).to(device)
    model.load_state_dict(model_state)
    model.eval()

    collate_fn = make_collate_fn(cfg)
    loader = DataLoader(
        Subset(dataset, test_indices.tolist() if hasattr(test_indices, "tolist") else test_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_task: list[np.ndarray] = []

    use_amp = cfg.use_amp and device.type == "cuda"
    for batch in loader:
        batch = batch_to_device(batch, device)
        with autocast("cuda", enabled=use_amp):
            h = model.encode_shared_features(batch["mel"])
        all_feats.append(h.float().cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        all_task.append(batch["task_id"].cpu().numpy())

    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    task_ids = np.concatenate(all_task)

    n = len(feats)
    if n < 4:
        logger.warning("t-SNE：样本少于 4，跳过")
        return

    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        feats, labels, task_ids = feats[idx], labels[idx], task_ids[idx]

    perplexity = min(30, max(5, len(feats) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=int(seed),
        init="pca",
        learning_rate="auto",
    )
    emb_2d = tsne.fit_transform(feats)

    fp = get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False
    title_fs, axis_fs, leg_fs, tick_fs = 28, 28, 26, 22
    t_fp = fp.copy()
    t_fp.set_size(title_fs)
    a_fp = fp.copy()
    a_fp.set_size(axis_fs)
    l_fp = fp.copy()
    l_fp.set_size(leg_fs)

    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    fig.patch.set_facecolor("white")

    task_specs = [
        (0, "dog_audio", "狗音频情绪特征分布"),
        (1, "cat_audio", "猫音频情绪特征分布"),
    ]
    cmap_dog = matplotlib.colormaps.get_cmap("tab10").resampled(
        TASK_META["dog_audio"]["num_classes"],
    )
    cmap_cat = matplotlib.colormaps.get_cmap("tab10").resampled(
        TASK_META["cat_audio"]["num_classes"],
    )
    cmaps = {"dog_audio": cmap_dog, "cat_audio": cmap_cat}

    for ax, (sp_id, tname, corner_txt) in zip(axes, task_specs):
        ax.set_facecolor("white")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.2)

        mask = task_ids == sp_id
        if not mask.any():
            ax.text(0.5, 0.5, "无样本", ha="center", va="center", fontproperties=fp, fontsize=22)
            continue

        class_names = TASK_META[tname]["classes"]
        cmap = cmaps[tname]
        for ci, cls in enumerate(class_names):
            cls_mask = mask & (labels == ci)
            if cls_mask.any():
                ax.scatter(
                    emb_2d[cls_mask, 0],
                    emb_2d[cls_mask, 1],
                    c=[cmap(ci)],
                    label=cls,
                    s=42,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.35,
                )
        # 狗：图例右上；猫：右下。缩小图例占地、略增散点图例标记
        if tname == "dog_audio":
            leg_loc = "upper right"
            leg_anchor = (0.98, 0.98)
            ncol = 1
        else:
            leg_loc = "lower right"
            leg_anchor = (0.98, 0.02)
            ncol = 2
        l_fp_small = fp.copy()
        l_fp_small.set_size(16)
        ax.legend(
            prop=l_fp_small, loc=leg_loc, framealpha=0.88, markerscale=2.1,
            bbox_to_anchor=leg_anchor,
            ncol=ncol, handlelength=0.9, handletextpad=0.35, labelspacing=0.28,
            borderpad=0.32,
        )
        ax.set_xlabel("t-SNE 1", fontproperties=a_fp)
        ax.set_ylabel("t-SNE 2", fontproperties=a_fp)
        ax.tick_params(labelsize=tick_fs)
        ax.margins(x=0.1, y=0.12)
        ax.set_title(corner_txt, loc="left", fontsize=title_fs, fontproperties=t_fp, pad=18)

    plt.tight_layout(pad=3.2)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("已保存 t-SNE 表征图：%s", save_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# Mel 频谱 Token-wise GradCAM
# ============================================================

def _get_mel_gradcam(
    model: CedAudioEmotionModel,
    mel: torch.Tensor,
    target_class: int,
    task_name: str,
    task_id: int,
    n_freq_patches: int = 4,
) -> np.ndarray:
    """
    对单个 Mel 样本计算 Token-wise GradCAM，返回 (n_freq_patches, n_time_patches) 热力图。

    原理：
      CED-Mini ViT 的 encoder 输出 token 序列 (1, N, D)，其中 N = n_freq_patches × n_time_patches。
      token 排列顺序为 (f, t) 展平（见 modeling_ced.py forward_features: flatten(2,3) + permute(0,2,1)）。
      对目标分类 logit 反向传播，取 token 维度梯度 × token 激活值，沿 D 维求和后 ReLU，
      reshape 到 (n_freq_patches, n_time_patches) 得到频谱注意力热力图。
    """
    model.eval()
    mel = mel.unsqueeze(0)  # (1, n_mels, T)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    # 挂钩在 encoder 最后一个 block 的输出（CedModel.blocks[-1]）
    target_block = model.encoder.blocks[-1]

    def fwd_hook(_, __, output: torch.Tensor) -> None:
        activations["feat"] = output  # (1, N, D)

    def bwd_hook(_, __, grad_out: tuple) -> None:
        gradients["feat"] = grad_out[0]  # (1, N, D)

    h_fwd = target_block.register_forward_hook(fwd_hook)
    h_bwd = target_block.register_full_backward_hook(bwd_hook)

    species_t = torch.tensor([task_id], device=mel.device)
    out = model(mel, species=species_t)
    logit = out[task_name][0, target_class]
    model.zero_grad()
    logit.backward()

    h_fwd.remove()
    h_bwd.remove()

    act = activations["feat"].squeeze(0)   # (N, D)
    grad = gradients["feat"].squeeze(0)    # (N, D)

    # 每个 token 的重要性 = (grad × act) 沿 D 维求和
    cam = (grad * act).sum(dim=-1)  # (N,)
    cam = torch.relu(cam)

    n_tokens = cam.shape[0]
    n_time_patches = n_tokens // n_freq_patches
    # 若 token 数不能整除（变长输入边界情况），截断对齐
    cam = cam[: n_freq_patches * n_time_patches]
    cam_2d = cam.reshape(n_freq_patches, n_time_patches).detach().cpu().float().numpy()

    # 归一化到 [0,1]
    vmax = cam_2d.max()
    if vmax > 1e-8:
        cam_2d = cam_2d / vmax
    return cam_2d


def _draw_gradcam_single_task(
    axes_row: list,
    task_name: str,
    task_id: int,
    test_indices: np.ndarray,
    dataset: "AudioEmotionDataset",
    model: "CedAudioEmotionModel",
    collate_fn,
    device: torch.device,
    N_FREQ_PATCHES: int,
    n_samples: int,
    rng: np.random.Generator,
    fp,
    title_fs: int,
    label_fs: int,
    tick_fs: int,
) -> None:
    """在给定的 axes_row（长度 = n_samples * 2）上绘制单任务 GradCAM。"""
    import torch.nn.functional as F  # noqa: PLC0415

    class_names = TASK_META[task_name]["classes"]
    task_indices = [idx for idx in test_indices if dataset.task_ids[idx] == task_id]
    if not task_indices:
        for ax in axes_row:
            ax.set_visible(False)
        return

    chosen = rng.choice(task_indices, size=min(n_samples, len(task_indices)), replace=False)

    for col_idx, sample_idx in enumerate(chosen):
        single_batch = collate_fn([dataset[sample_idx]])
        single_batch = batch_to_device(single_batch, device)
        mel = single_batch["mel"][0]
        true_label = int(single_batch["label"][0].item())
        mel_np = mel.detach().cpu().float().numpy()

        with torch.enable_grad():
            cam_2d = _get_mel_gradcam(model, mel, true_label, task_name, task_id, N_FREQ_PATCHES)

        cam_tensor = torch.from_numpy(cam_2d).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(
            cam_tensor, size=(mel_np.shape[0], mel_np.shape[1]),
            mode="bilinear", align_corners=False,
        ).squeeze().numpy()

        # 左图：原始 Mel
        ax_mel = axes_row[col_idx * 2]
        ax_mel.imshow(mel_np, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
        ax_mel.set_xlabel("时间帧", fontproperties=fp, fontsize=label_fs)
        ax_mel.set_ylabel("Mel 频带", fontproperties=fp, fontsize=label_fs)
        ax_mel.tick_params(labelsize=tick_fs)
        ax_mel.set_facecolor("white")
        for s in ax_mel.spines.values():
            s.set_linewidth(1.2)
        ax_mel.set_title(
            f"{TASK_ZH.get(task_name, task_name)}\n真实: {class_names[true_label]}",
            fontproperties=_title_font_prop(fp, title_fs), pad=14,
        )

        # 右图：GradCAM 叠加
        ax_cam = axes_row[col_idx * 2 + 1]
        mel_norm = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)
        cam_colored = plt.cm.jet(cam_resized)[..., :3]
        mel_rgb = plt.cm.magma(mel_norm)[..., :3]
        overlay = np.clip(0.5 * mel_rgb + 0.5 * cam_colored, 0, 1)
        ax_cam.imshow(overlay, aspect="auto", origin="lower", interpolation="nearest")
        ax_cam.set_xlabel("时间帧", fontproperties=fp, fontsize=label_fs)
        ax_cam.set_ylabel("Mel 频带", fontproperties=fp, fontsize=label_fs)
        ax_cam.tick_params(labelsize=tick_fs)
        ax_cam.set_facecolor("white")
        for s in ax_cam.spines.values():
            s.set_linewidth(1.2)
        ax_cam.set_title("Mel GradCAM", fontproperties=_title_font_prop(fp, title_fs), pad=14)

    # 隐藏多余子图
    for col_idx in range(len(chosen), n_samples):
        axes_row[col_idx * 2].set_visible(False)
        axes_row[col_idx * 2 + 1].set_visible(False)


def plot_mel_gradcam_samples(
    model_state: dict,
    dataset: "AudioEmotionDataset",
    test_indices: np.ndarray,
    device: torch.device,
    cfg: "TrainConfig",
    save_path: Path,
    n_samples_per_task: int = 4,
    seed: int = 42,
) -> None:
    """
    对测试集随机样本绘制 Mel 频谱原图 + GradCAM 叠加图。
    每个任务（狗/猫）单独保存一张图，避免 16 个子图过于紧凑。

    save_path 作为基础路径，实际保存为：
      {stem}_dog{suffix}  和  {stem}_cat{suffix}
    """
    rng = np.random.default_rng(seed)
    fp = get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    model = CedAudioEmotionModel(cfg).to(device)
    model.load_state_dict(model_state)
    model.eval()

    N_FREQ_PATCHES = cfg.n_mels // 16
    collate_fn = make_collate_fn(cfg)

    _title_fs = 24
    _label_fs = 20
    _tick_fs = 18

    # 任务名 → 文件后缀映射
    task_suffix = {"dog_audio": "_dog", "cat_audio": "_cat"}
    stem = save_path.stem
    suffix = save_path.suffix

    for task_name in TASK_HEADS:
        task_id = list(TASK_HEADS).index(task_name)
        n_cols = n_samples_per_task * 2
        fig, axes = plt.subplots(1, n_cols, figsize=(7.5 * n_cols, 8.0))
        fig.patch.set_facecolor("white")
        axes_row = list(axes)

        _draw_gradcam_single_task(
            axes_row=axes_row,
            task_name=task_name,
            task_id=task_id,
            test_indices=test_indices,
            dataset=dataset,
            model=model,
            collate_fn=collate_fn,
            device=device,
            N_FREQ_PATCHES=N_FREQ_PATCHES,
            n_samples=n_samples_per_task,
            rng=rng,
            fp=fp,
            title_fs=_title_fs,
            label_fs=_label_fs,
            tick_fs=_tick_fs,
        )

        plt.tight_layout(pad=3.2)
        out_path = save_path.parent / f"{stem}{task_suffix.get(task_name, f'_{task_name}')}{suffix}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("已保存 Mel GradCAM 可视化图（%s）：%s", task_name, out_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
