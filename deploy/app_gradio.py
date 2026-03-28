# -*- coding: utf-8 -*-
"""
宠物多模态情绪识别 —— Gradio Web UI

基于 deploy/try1.py 的融合推理引擎，提供可视化交互界面。
支持：图像+音频 / 仅图像 / 仅音频 三种输入模式。

运行：
  conda activate d2l
  python deploy/app_gradio.py

若 7860 已被占用（例如上次 Gradio 未关闭），脚本会自动顺延到下一个空闲端口；
也可手动指定起始端口：set GRADIO_SERVER_PORT=7870 （Windows PowerShell: $env:GRADIO_SERVER_PORT=7870）
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 将项目自带的 ffmpeg_bin（含 ffmpeg.exe + ffprobe.exe）注入 PATH
# 确保 Gradio / pydub 音频组件能正确调用 ffprobe
_ffmpeg_bin_dir = str(ROOT / "ffmpeg_bin")
if Path(_ffmpeg_bin_dir).is_dir() and _ffmpeg_bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from deploy.try1 import (
    MultimodalPetEmotionSystem,
    FusionResult,
    DOG_IMG_CLASSES,
    CAT_IMG_CLASSES,
    DOG_AUDIO_CLASSES,
    CAT_AUDIO_CLASSES,
    DOG_IMG_DESC,
    CAT_IMG_DESC,
    DOG_AUDIO_DESC,
    CAT_AUDIO_DESC,
)

# ════════════════════════════════════════════════════════════════
# 字体配置
# ════════════════════════════════════════════════════════════════
_FONT_CANDIDATES = [
    "SimHei", "Microsoft YaHei", "Source Han Sans CN",
    "WenQuanYi Micro Hei", "Noto Sans CJK SC",
]

def _get_cn_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in available:
            return name
    return None

_CN_FONT = _get_cn_font()
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ════════════════════════════════════════════════════════════════
# 全局模型（延迟加载）
# ════════════════════════════════════════════════════════════════
_system: MultimodalPetEmotionSystem | None = None


def _get_system() -> MultimodalPetEmotionSystem:
    global _system
    if _system is None:
        _system = MultimodalPetEmotionSystem(device="auto")
    return _system


# ════════════════════════════════════════════════════════════════
# 可视化：统一状态分布柱状图
# ════════════════════════════════════════════════════════════════
_STATE_COLORS = {
    "威胁/攻击": "#E74C3C",
    "积极/兴奋": "#2ECC71",
    "放松/平静": "#3498DB",
    "低落/不安": "#9B59B6",
    "攻击/防御": "#E74C3C",
    "恐惧/警戒": "#E67E22",
    "积极/亲和": "#2ECC71",
    "平静/中性": "#3498DB",
    "低落/不适": "#9B59B6",
}

# 图表中仅用英文展示统一状态（与中文键一一对应，供 bar 配色与映射使用）
UNIFIED_STATE_EN: dict[str, str] = {
    "威胁/攻击": "Threat / aggression",
    "积极/兴奋": "Positive / excited",
    "放松/平静": "Relaxed / calm",
    "低落/不安": "Low / distressed",
    "攻击/防御": "Attack / defence",
    "恐惧/警戒": "Fear / alert",
    "积极/亲和": "Positive / affiliative",
    "平静/中性": "Neutral / calm",
    "低落/不适": "Low / discomfort",
}


def _plot_unified_distribution(dist: dict[str, float]) -> plt.Figure:
    names_cn = list(dist.keys())
    values = list(dist.values())
    labels_en = [UNIFIED_STATE_EN.get(n, n) for n in names_cn]
    colors = [_STATE_COLORS.get(n, "#95A5A6") for n in names_cn]

    fig, ax = plt.subplots(figsize=(9.5, max(2.8, len(names_cn) * 0.72 + 1.0)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(names_cn))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left", fontsize=18, fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_en, fontsize=16)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probability", fontsize=18)
    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=14)

    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.32, right=0.95)
    return fig


# ════════════════════════════════════════════════════════════════
# 可视化：原始模型 softmax 分布柱状图
# ════════════════════════════════════════════════════════════════

def _display_class_en(name: str) -> str:
    """类别名在图上显示为易读的英文（骆驼拼写法拆成带空格）。"""
    if not name:
        return name
    # 连续大小写变化处拆词：HuntingMind -> Hunting Mind
    out: list[str] = []
    for i, ch in enumerate(name):
        if i > 0 and ch.isupper() and name[i - 1].islower():
            out.append(" ")
        out.append(ch)
    return "".join(out)


def _plot_raw_softmax(classes: list[str], probs: list[float], color: str) -> plt.Figure:
    """横向柱状图：仅英文标签，避免纵轴多类英文名重叠。"""
    labels = [_display_class_en(c) for c in classes]
    n = len(classes)
    # 每行 0.72 英寸 + 顶底各留 0.6 英寸，确保第一行不被标题遮住
    fig_h = max(3.5, n * 0.72 + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, probs, color=color, alpha=0.85, height=0.52, edgecolor="white")

    for bar, val in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left", fontsize=15, fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlim(0, min(1.18, max(probs) * 1.3 + 0.10))
    ax.set_xlabel("Probability", fontsize=16)
    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)

    # 顶部额外留白，防止第一个条形被 Gradio 标题栏遮挡
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.18, right=0.95)
    return fig


# ════════════════════════════════════════════════════════════════
# 可视化：融合权重饼图
# ════════════════════════════════════════════════════════════════

def _plot_weights_pie(weights: dict[str, float]) -> plt.Figure:
    key_to_en = {"图像": "Image", "音频": "Audio"}
    labels = [key_to_en.get(k, k) for k in weights.keys()]
    sizes = list(weights.values())
    colors_pie = ["#3498DB", "#E67E22"]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors_pie, startangle=90,
        textprops={"fontsize": 18},
        pctdistance=0.6,
    )
    for t in autotexts:
        t.set_fontsize(18)
        t.set_fontweight("bold")

    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# 推理回调
# ════════════════════════════════════════════════════════════════

def _risk_emoji(level: str) -> str:
    return {"警惕": "🔴", "关注": "🟡", "正常": "🟢"}.get(level, "⚪")


def run_inference(image, audio, species_hint):
    if image is None and audio is None:
        gr.Warning("请至少上传一张图像或一段音频")
        return (None,) * 7

    system = _get_system()

    species_map = {"自动检测": None, "狗": "狗", "猫": "猫"}
    hint = species_map.get(species_hint)

    img_path = image if image is not None else None
    audio_path = audio if audio is not None else None

    try:
        result: FusionResult = system.infer(
            image_path=img_path,
            audio_path=audio_path,
            species_hint=hint,
        )
    except Exception as e:
        gr.Warning(f"推理出错: {e}")
        return (None,) * 7

    from deploy.try1 import _softmax

    is_dog = (result.species == "狗")

    # --- 1. 核心信息 Markdown ---
    risk_icon = _risk_emoji(result.risk_level)
    special_line = ""
    if result.special_behavior:
        special_line = f"| **特殊行为** | ⚠️ {result.special_behavior} |\n"

    consistency_text = result.consistency
    if result.consistency_score is not None:
        consistency_text += f" ({result.consistency_score:.2f})"

    summary_md = f"""
## 🐾 识别结果

| 项目 | 结果 |
|:-----|:-----|
| **物种** | {result.species} ({result.species_confidence:.1%}) |
| **综合状态** | **{result.primary_state}** ({result.primary_confidence:.1%}) |
| **风险等级** | {risk_icon} {result.risk_level} |
| **模态一致性** | {consistency_text} |
{special_line}

---

### 综合解释

{result.explanation}
"""

    # --- 2. 统一状态分布图 ---
    fig_unified = _plot_unified_distribution(result.unified_distribution)

    # --- 3. 图像侧原始分布图（利用 FusionResult 中已有的置信度重建近似分布） ---
    fig_img = None
    if result.image_prediction is not None and img_path is not None:
        classes = DOG_IMG_CLASSES if is_dog else CAT_IMG_CLASSES
        img_logits = system._infer_image(img_path)
        key = "dog_img" if is_dog else "cat_img"
        probs = _softmax(img_logits[key]).tolist()
        fig_img = _plot_raw_softmax(classes, probs, "#3498DB")

    # --- 4. 音频侧原始分布图 ---
    fig_audio = None
    if result.audio_prediction is not None and audio_path is not None:
        classes = DOG_AUDIO_CLASSES if is_dog else CAT_AUDIO_CLASSES
        audio_logits = system._infer_audio(audio_path)
        key = "dog_audio" if is_dog else "cat_audio"
        probs = _softmax(audio_logits[key]).tolist()
        fig_audio = _plot_raw_softmax(classes, probs, "#E67E22")

    # --- 5. 融合权重饼图 ---
    fig_pie = _plot_weights_pie(result.modality_weights)

    # --- 6. 证据摘要 Markdown ---
    evidence_parts = []
    if result.image_prediction is not None:
        img_desc_all = {**DOG_IMG_DESC, **CAT_IMG_DESC}
        evidence_parts.append(
            f"**视觉证据：** {result.image_prediction}"
            f" ({img_desc_all.get(result.image_prediction, '')}) "
            f"— 置信度 {result.image_confidence:.1%}"
        )
    if result.audio_prediction is not None:
        audio_desc_all = {**DOG_AUDIO_DESC, **CAT_AUDIO_DESC}
        evidence_parts.append(
            f"**声学证据：** {result.audio_prediction}"
            f" ({audio_desc_all.get(result.audio_prediction, '')}) "
            f"— 置信度 {result.audio_confidence:.1%}"
        )

    evidence_md = "\n\n".join(evidence_parts) if evidence_parts else ""

    return (
        summary_md,       # 核心信息
        fig_unified,      # 统一状态分布
        fig_img,          # 图像原始分布
        fig_audio,        # 音频原始分布
        fig_pie,          # 融合权重饼图
        evidence_md,      # 证据摘要
        result.explanation,  # 纯文本解释（用于日志）
    )


# ════════════════════════════════════════════════════════════════
# Gradio 界面
# ════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="宠物多模态情绪识别系统",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange",
            font=[gr.themes.GoogleFont("Noto Sans SC"), "sans-serif"],
        ),
    ) as app:
        gr.Markdown(
            """
# 🐾 宠物多模态情绪识别系统
### 基于 DINOv3-ConvNeXt-Tiny 视觉模型 + CED-Mini 声学模型的决策级语义映射融合

上传宠物的**图像**和/或**音频**，系统将通过跨模态语义融合分析宠物的情绪状态。
支持三种模式：**图像+音频**（多模态融合）、**仅图像**、**仅音频**。
""",
        )

        with gr.Row():
            # ── 左侧：输入区 ──
            with gr.Column(scale=2):
                gr.Markdown("### 输入")
                img_input = gr.Image(
                    label="上传宠物图像",
                    type="filepath",
                    height=280,
                )
                audio_input = gr.Audio(
                    label="上传宠物音频",
                    type="filepath",
                )
                species_dropdown = gr.Dropdown(
                    choices=["自动检测", "狗", "猫"],
                    value="自动检测",
                    label="物种提示（可选）",
                )
                run_btn = gr.Button(
                    "🔍 开始识别",
                    variant="primary",
                    size="lg",
                )

            # ── 右侧：结果区 ──
            with gr.Column(scale=3):
                gr.Markdown("### 识别结果")
                summary_output = gr.Markdown(
                    value="*等待输入...*",
                    label="核心结果",
                )
                evidence_output = gr.Markdown(
                    value="",
                    label="模态证据",
                )

        gr.Markdown("---")
        gr.Markdown("### 详细分析")

        with gr.Row():
            unified_plot = gr.Plot(label="统一状态分布")
            weights_plot = gr.Plot(label="融合权重")

        with gr.Row():
            img_plot = gr.Plot(label="图像模型原始分布")
            audio_plot = gr.Plot(label="音频模型原始分布")

        explanation_state = gr.State(value="")

        run_btn.click(
            fn=run_inference,
            inputs=[img_input, audio_input, species_dropdown],
            outputs=[
                summary_output,
                unified_plot,
                img_plot,
                audio_plot,
                weights_plot,
                evidence_output,
                explanation_state,
            ],
        )

        gr.Markdown(
            """
---
<center>

**技术栈：** DINOv3-ConvNeXt-Tiny (图像) · CED-Mini (音频) · 行为学语义映射融合

**融合策略：** 统一状态空间 + 软映射矩阵 + 置信度自适应加权

</center>
""",
        )

    return app


def _find_free_port(start: int = 7860, span: int = 100) -> int:
    """从 start 起尝试绑定 TCP（0.0.0.0），返回第一个可用端口。"""
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(
        f"在 {start}～{start + span - 1} 内无空闲端口，请关闭占用进程后再试。"
    )


# ════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_app()
    preferred = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    port = _find_free_port(start=preferred, span=100)
    if port != preferred:
        print(f"[提示] 端口 {preferred} 已被占用，顺延至 {port}。")
    print(f"[启动] 本地访问地址：http://localhost:{port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=True,
    )
