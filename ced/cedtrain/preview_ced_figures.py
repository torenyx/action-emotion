# -*- coding: utf-8 -*-
"""
CED 可解释性组图快速预览（不训练、不加载模型）

用途：用随机合成的概率与混淆矩阵调用与训练后相同的绘图函数，
     先看版式/字号/图例是否满意，再跑完整训练。

P/R/F1 柱状图等与正式训练共用 `cedtrain.mlp_interpretability` 内实现
（如 `plot_per_class_prf_bars`）；改图例位置/字号请编辑该模块，本脚本仅负责合成数据与调用。

运行（勿用 base 环境）：
  conda activate d2l
  python cedtrain/preview_ced_figures.py

输出：
  figure/preview_ced_*_{时间戳}.png
  txt/preview_ced_figures_{时间戳}.txt（简要日志）
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cedtrain.config import FIG_DIR, TASK_META, TXT_DIR
from cedtrain.mlp_interpretability import run_mlp_interpretability_suite
from cedtrain.utils import get_zh_font
from cedtrain.visualization import plot_confusion_matrices


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def make_synthetic_prob_bundle(
    seed: int = 42,
    n_dog: int = 120,
    n_cat: int = 200,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[str]]]:
    """合成 (probs, targets)，形状与真实评估一致。"""
    rng = np.random.default_rng(seed)
    task_class_names = {k: list(v["classes"]) for k, v in TASK_META.items()}

    def one_task(task_key: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        n_cls = TASK_META[task_key]["num_classes"]
        targets = rng.integers(0, n_cls, size=n)
        logits = rng.normal(0, 1.2, size=(n, n_cls))
        for i in range(n):
            logits[i, targets[i]] += rng.uniform(1.2, 2.8)  # 倾向预测对
        # 再故意错一部分
        flip = rng.random(n) < 0.12
        wrong = rng.integers(0, n_cls, size=n)
        for i in np.where(flip)[0]:
            if wrong[i] != targets[i]:
                targets[i] = wrong[i]
            logits[i, targets[i]] += 0.8
        probs = _softmax(logits.astype(np.float64), axis=1)
        return probs.astype(np.float32), targets.astype(np.int64)

    prob_bundle = {
        "dog_audio": one_task("dog_audio", n_dog),
        "cat_audio": one_task("cat_audio", n_cat),
    }
    return prob_bundle, task_class_names


def make_synthetic_cm_data(seed: int = 7) -> dict[str, tuple[np.ndarray, list[str]]]:
    """合成混淆矩阵（含非对角错误，便于 Top 混淆条形图有内容）。"""
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[np.ndarray, list[str]]] = {}

    # 狗 4 类：对角偏大，少量混淆
    names_d = TASK_META["dog_audio"]["classes"]
    cm_d = np.zeros((4, 4), dtype=np.int64)
    for i in range(4):
        cm_d[i, i] = int(rng.integers(22, 35))
    cm_d[0, 2] = 3
    cm_d[2, 0] = 2
    cm_d[1, 3] = 1
    out["dog_audio"] = (cm_d, list(names_d))

    # 猫 10 类：多组非对角
    names_c = TASK_META["cat_audio"]["classes"]
    n_c = 10
    cm_c = np.zeros((n_c, n_c), dtype=np.int64)
    for i in range(n_c):
        cm_c[i, i] = int(rng.integers(15, 32))
    cm_c[3, 8] = 6  # Happy -> Paining
    cm_c[8, 3] = 5
    cm_c[0, 1] = 4
    cm_c[2, 0] = 3
    cm_c[4, 7] = 2
    cm_c[7, 4] = 2
    out["cat_audio"] = (cm_c, list(names_c))

    return out


def plot_tsne_preview_fake(save_path: Path, seed: int = 42) -> None:
    """
    仅用随机二维簇模拟 t-SNE 散点布局（与正式脚本的配色/图例位置一致），无 TSNE 计算、无模型。
    """
    rng = np.random.default_rng(seed)
    fp = get_zh_font()
    plt.rcParams["axes.unicode_minus"] = False

    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    task_ids: list[np.ndarray] = []

    # 狗：4 个类簇
    dog_centers = np.array([[0, 8], [10, -4], [-8, -6], [12, 10]], dtype=np.float64)
    for ci in range(4):
        pts = dog_centers[ci] + rng.normal(0, 1.1, size=(45, 2))
        points.append(pts)
        labels.append(np.full(45, ci, dtype=np.int64))
        task_ids.append(np.zeros(45, dtype=np.int64))

    # 猫：10 个类簇
    cat_centers = rng.uniform(-18, 22, size=(10, 2))
    for ci in range(10):
        pts = cat_centers[ci] + rng.normal(0, 0.95, size=(35, 2))
        points.append(pts)
        labels.append(np.full(35, ci, dtype=np.int64))
        task_ids.append(np.ones(35, dtype=np.int64))

    emb_2d = np.vstack(points)
    labels = np.concatenate(labels)
    task_ids = np.concatenate(task_ids)

    import matplotlib as mpl

    title_fs, axis_fs, tick_fs = 28, 28, 22
    t_fp = fp.copy()
    t_fp.set_size(title_fs)
    a_fp = fp.copy()
    a_fp.set_size(axis_fs)

    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    fig.patch.set_facecolor("white")

    task_specs = [
        (0, "dog_audio", "狗音频情绪特征分布（预览·合成簇）"),
        (1, "cat_audio", "猫音频情绪特征分布（预览·合成簇）"),
    ]
    cmap_dog = mpl.colormaps.get_cmap("tab10").resampled(TASK_META["dog_audio"]["num_classes"])
    cmap_cat = mpl.colormaps.get_cmap("tab10").resampled(TASK_META["cat_audio"]["num_classes"])
    cmaps = {"dog_audio": cmap_dog, "cat_audio": cmap_cat}

    for ax, (sp_id, tname, corner_txt) in zip(axes, task_specs):
        ax.set_facecolor("white")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(1.2)

        mask = task_ids == sp_id
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
        if tname == "dog_audio":
            leg_loc, leg_anchor, ncol = "upper right", (0.98, 0.98), 1
        else:
            leg_loc, leg_anchor, ncol = "lower right", (0.98, 0.02), 2
        l_fp_small = fp.copy()
        l_fp_small.set_size(16)
        ax.legend(
            prop=l_fp_small, loc=leg_loc, framealpha=0.88, markerscale=2.1,
            bbox_to_anchor=leg_anchor, ncol=ncol,
            handlelength=0.9, handletextpad=0.35, labelspacing=0.28, borderpad=0.32,
        )
        ax.set_xlabel("t-SNE 1", fontproperties=a_fp)
        ax.set_ylabel("t-SNE 2", fontproperties=a_fp)
        ax.tick_params(labelsize=tick_fs)
        ax.margins(x=0.1, y=0.12)
        ax.set_title(corner_txt, loc="left", fontsize=title_fs, fontproperties=t_fp, pad=18)

    plt.tight_layout(pad=3.2)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    ts = _timestamp()
    prefix = "preview_ced_"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    log_buf = io.StringIO()

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg)

    logging.basicConfig(stream=log_buf, level=logging.INFO, format="%(message)s")

    log("CED 组图预览（合成数据，无训练）")
    log(f"时间戳: {ts}")

    prob_bundle, task_class_names = make_synthetic_prob_bundle(seed=42)
    cm_data = make_synthetic_cm_data(seed=7)

    cm_path = FIG_DIR / f"{prefix}confusion_matrix_{ts}.png"
    plot_confusion_matrices(cm_data, cm_path)
    log(f"混淆矩阵: {cm_path}")

    # 含 interpret_prf：与 run_train 相同，样式在 mlp_interpretability.plot_per_class_prf_bars
    run_mlp_interpretability_suite(
        cm_data, prob_bundle, task_class_names, FIG_DIR, ts, prefix=prefix,
    )
    log(f"P/R/F1、ROC、置信度、Top混淆: {prefix}interpret_*_{ts}.png")

    tsne_path = FIG_DIR / f"{prefix}tsne_{ts}.png"
    plot_tsne_preview_fake(tsne_path, seed=42)
    log(f"t-SNE 布局预览: {tsne_path}")

    log("")
    log("说明：GradCAM 需真实模型与前向/反向，本脚本不生成；训练后见 ced_mel_gradcam_*_dog/cat_*.png")

    extra = log_buf.getvalue()
    if extra.strip():
        log_lines.append("--- logging ---")
        log_lines.append(extra.strip())

    txt_path = TXT_DIR / f"preview_ced_figures_{ts}.txt"
    txt_path.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"日志: {txt_path}")


if __name__ == "__main__":
    main()
