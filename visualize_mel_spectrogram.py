# -*- coding: utf-8 -*-
"""
梅尔频谱可视化调试工具

功能：
  1. 狗 growling / 猫 Angry 各随机抽取 4 个样本，绘制 1×4 梅尔频谱图
  2. 狗 growling / 猫 Angry 对比波形图 + 梅尔频谱图（各 1 样本，左波形右频谱）
  3. 狗/猫各类别频带能量轮廓对比（均值 ± std 曲线）

输出：
  figure/mel_spec_growling_1x4_{时间戳}.png
  figure/mel_spec_angry_1x4_{时间戳}.png
  figure/mel_spec_waveform_compare_{时间戳}.png
  figure/mel_spec_band_energy_{时间戳}.png
  txt/mel_spec_log_{时间戳}.txt
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# ============================================================
# 路径配置（与 cedtrain/config.py 完全对齐）
# ============================================================
DOG_AUDIO_DIR = ROOT / "data" / "Pet dog sound event"
CAT_AUDIO_DIR = ROOT / "data" / "CatSound_DataSet_V2" / "NAYA_DATA_AUG1X"
FIG_DIR       = ROOT / "figure"
TXT_DIR       = ROOT / "txt"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# ============================================================
# CED-Mini 梅尔参数（与 cedtrain/config.py 严格一致）
# ============================================================
SR         = 16000
N_FFT      = 512
WIN_SIZE   = 512
HOP_SIZE   = 160
N_MELS     = 64
F_MIN      = 0
F_MAX      = 8000
MEL_TOP_DB = 120
MAX_SEC    = 10.0
CLIP_SEC   = 10.0

# 每类最多显示的样本数
SAMPLES_PER_CLASS = 4
# 频带能量箱线图：每类随机抽取的文件数
N_BOX_SAMPLES = 30
# 随机种子
SEED = 42

TS = datetime.now().strftime("%Y%m%d%H%M%S")

# ============================================================
# 日志
# ============================================================
log_path = TXT_DIR / f"mel_spec_log_{TS}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ],
)
logger = logging.getLogger("mel_vis")


# ============================================================
# 字体
# ============================================================
def _get_zh_font():
    """返回可用的中文字体属性对象，优先使用 SimHei。"""
    from matplotlib import font_manager as fm
    candidates = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                  "Noto Sans CJK SC", "Arial Unicode MS"]
    for name in candidates:
        if fm.findfont(fm.FontProperties(family=name)) != fm.findfont(fm.FontProperties()):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            logger.info("使用中文字体：%s", name)
            return fm.FontProperties(family=name)
    plt.rcParams["axes.unicode_minus"] = False
    logger.warning("未找到中文字体，使用默认字体")
    return fm.FontProperties()


# ============================================================
# 音频工具
# ============================================================
_RESAMPLE_CACHE: dict[tuple[int, int], T.Resample] = {}

def _get_resample(orig_sr: int, target_sr: int) -> T.Resample:
    key = (orig_sr, target_sr)
    if key not in _RESAMPLE_CACHE:
        _RESAMPLE_CACHE[key] = T.Resample(orig_sr, target_sr)
    return _RESAMPLE_CACHE[key]


def load_waveform(path: Path) -> torch.Tensor:
    """读取 → 重采样 16kHz → 单声道 → 峰值归一化 → loop padding/截断。"""
    waveform, sr = torchaudio.load(str(path))
    if sr != SR:
        waveform = _get_resample(sr, SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    t_len = waveform.shape[1]
    if t_len == 0:
        raise ValueError(f"空音频: {path}")
    max_samples = int(MAX_SEC * SR)
    if t_len > max_samples:
        waveform = waveform[:, :max_samples]
        t_len = max_samples
    peak = waveform.abs().max().clamp(min=1e-8)
    waveform = waveform / peak
    clip_samples = int(CLIP_SEC * SR)
    if t_len < clip_samples:
        repeats = (clip_samples + t_len - 1) // t_len
        waveform = torch.tile(waveform, (1, repeats))
        waveform = waveform[:, :clip_samples]
    return waveform


def build_mel_transform():
    mel_spec = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN_SIZE,
        hop_length=HOP_SIZE, f_min=F_MIN, f_max=F_MAX,
        n_mels=N_MELS, center=True,
    )
    amp_to_db = T.AmplitudeToDB(top_db=MEL_TOP_DB)
    return mel_spec, amp_to_db


def waveform_to_mel(waveform: torch.Tensor,
                    mel_spec: T.MelSpectrogram,
                    amp_to_db: T.AmplitudeToDB) -> np.ndarray:
    """返回 (n_mels, n_frames) numpy 数组，值为 log-mel dB。"""
    mel = mel_spec(waveform)
    mel = amp_to_db(mel)
    return mel.squeeze(0).numpy()   # (64, T)


# ============================================================
# 数据扫描
# ============================================================
def scan_classes(root: Path) -> dict[str, list[Path]]:
    """扫描 root/<class>/ 目录，返回 {class_name: [file_paths]}"""
    result: dict[str, list[Path]] = {}
    if not root.exists():
        logger.warning("目录不存在：%s", root)
        return result
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [f for f in sorted(class_dir.iterdir())
                 if f.suffix.lower() in AUDIO_EXTENSIONS]
        if files:
            result[class_dir.name] = files
    return result


# ============================================================
# 绘图 1：单类别 1×4 梅尔频谱图
# ============================================================
def plot_mel_1x4(
    files: list[Path],
    mel_spec: T.MelSpectrogram,
    amp_to_db: T.AmplitudeToDB,
    fp,
    cls_name: str,
    species_cn: str,
    out_key: str,
) -> Path:
    """从 files 中随机取 4 个样本，绘制 1 行 4 列梅尔频谱图。"""
    rng = random.Random(SEED)
    chosen = rng.sample(files, min(4, len(files)))
    while len(chosen) < 4:
        chosen.append(chosen[-1])

    label_fs = 18
    tick_fs  = 14
    color_map = "magma"

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor="white")

    for c, fpath in enumerate(chosen):
        ax = axes[c]
        try:
            waveform = load_waveform(fpath)
            mel_data = waveform_to_mel(waveform, mel_spec, amp_to_db)
        except Exception as e:
            logger.warning("跳过 %s：%s", fpath.name, e)
            ax.axis("off")
            continue

        im = ax.imshow(
            mel_data, aspect="auto", origin="lower",
            cmap=color_map, interpolation="nearest",
            vmin=-MEL_TOP_DB, vmax=0,
        )
        ax.set_title(f"样本 {c + 1}", fontproperties=fp, fontsize=label_fs)
        ax.set_xlabel("时间帧", fontproperties=fp, fontsize=label_fs - 1)
        if c == 0:
            ax.set_ylabel("梅尔频带", fontproperties=fp, fontsize=label_fs - 1)
        ax.tick_params(labelsize=tick_fs)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    # 右侧 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.68])
    sm = plt.cm.ScalarMappable(
        cmap=color_map, norm=plt.Normalize(vmin=-MEL_TOP_DB, vmax=0)
    )
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("能量 (dB)", fontproperties=fp, fontsize=label_fs - 2)
    cbar.ax.tick_params(labelsize=tick_fs - 1)

    # 顶部注释
    fig.text(
        0.46, 0.97,
        f"{species_cn} · {cls_name}  梅尔频谱（64 mel, log-dB）",
        ha="center", va="top",
        fontproperties=fp, fontsize=label_fs,
    )

    plt.subplots_adjust(left=0, right=0.91, top=0.90, bottom=0.08, wspace=0.05)
    out = FIG_DIR / f"mel_spec_{out_key}_1x4_{TS}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("已保存：%s", out)
    return out


# ============================================================
# 绘图 2：growling vs Angry 波形 + 梅尔频谱对比（2 行 × 2 列）
# ============================================================
def plot_waveform_vs_mel(
    dog_growling_files: list[Path],
    cat_angry_files: list[Path],
    mel_spec: T.MelSpectrogram,
    amp_to_db: T.AmplitudeToDB,
    fp,
) -> Path:
    rng = random.Random(SEED)
    samples = [
        ("狗", "growling", rng.choice(dog_growling_files)),
        ("猫", "Angry",    rng.choice(cat_angry_files)),
    ]

    label_fs = 18
    tick_fs  = 14

    fig, axes = plt.subplots(2, 2, figsize=(18, 9), facecolor="white")

    for row, (sp_cn, cls_name, fpath) in enumerate(samples):
        waveform = load_waveform(fpath)
        mel_data = waveform_to_mel(waveform, mel_spec, amp_to_db)
        wv_np = waveform.squeeze(0).numpy()
        duration_s = len(wv_np) / SR

        # 左列：波形
        ax_wv = axes[row, 0]
        t = np.linspace(0, duration_s, len(wv_np))
        ax_wv.plot(t, wv_np, color="#2563EB", linewidth=0.5, alpha=0.85)
        ax_wv.set_xlabel("时间 (s)", fontproperties=fp, fontsize=label_fs - 1)
        ax_wv.set_ylabel("幅度", fontproperties=fp, fontsize=label_fs - 1)
        ax_wv.set_title(f"{sp_cn} · {cls_name} — 波形",
                        fontproperties=fp, fontsize=label_fs)
        ax_wv.set_xlim(0, duration_s)
        ax_wv.tick_params(labelsize=tick_fs)
        ax_wv.set_facecolor("white")
        for spine in ax_wv.spines.values():
            spine.set_edgecolor("#cccccc")

        # 右列：梅尔频谱
        ax_mel = axes[row, 1]
        im = ax_mel.imshow(
            mel_data, aspect="auto", origin="lower",
            cmap="magma", interpolation="nearest",
            vmin=-MEL_TOP_DB, vmax=0,
            extent=[0, duration_s, 0, F_MAX / 1000],
        )
        ax_mel.set_xlabel("时间 (s)", fontproperties=fp, fontsize=label_fs - 1)
        ax_mel.set_ylabel("频率 (kHz)  [Mel 轴]", fontproperties=fp, fontsize=label_fs - 1)
        ax_mel.set_title(f"{sp_cn} · {cls_name} — 梅尔频谱 (64 mel, log-dB)",
                         fontproperties=fp, fontsize=label_fs)
        ax_mel.tick_params(labelsize=tick_fs)
        ax_mel.set_facecolor("white")
        for spine in ax_mel.spines.values():
            spine.set_edgecolor("#cccccc")
        cbar = plt.colorbar(im, ax=ax_mel, pad=0.02)
        cbar.set_label("能量 (dB)", fontproperties=fp, fontsize=tick_fs)
        cbar.ax.tick_params(labelsize=tick_fs - 1)

    plt.tight_layout(pad=2.5)
    out = FIG_DIR / f"mel_spec_waveform_compare_{TS}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("已保存：%s", out)
    return out
def plot_band_energy_boxplot(
    all_class_files: dict[str, dict[str, list[Path]]],
    mel_spec: T.MelSpectrogram,
    amp_to_db: T.AmplitudeToDB,
    fp,
    n_per_class: int = N_BOX_SAMPLES,
) -> Path:
    rng = random.Random(SEED)

    # 对每个样本计算各 mel 频带的时间平均能量（标量数组，长 64）
    species_band_data: dict[str, dict[str, np.ndarray]] = {}  # sp -> cls -> (N, 64)

    for species, class_files in all_class_files.items():
        sp_data: dict[str, np.ndarray] = {}
        for cls_name, files in class_files.items():
            chosen = rng.sample(files, min(n_per_class, len(files)))
            band_vecs = []
            for fp_audio in chosen:
                try:
                    waveform = load_waveform(fp_audio)
                    mel_data = waveform_to_mel(waveform, mel_spec, amp_to_db)
                    # 每个频带的时间均值 → (64,)
                    band_vecs.append(mel_data.mean(axis=1))
                except Exception:
                    pass
            if band_vecs:
                sp_data[cls_name] = np.stack(band_vecs, axis=0)   # (N, 64)
        species_band_data[species] = sp_data

    all_species = list(species_band_data.keys())
    n_species = len(all_species)

    fig, axes = plt.subplots(1, n_species, figsize=(max(n_species * 10, 14), 7),
                             facecolor="white")
    if n_species == 1:
        axes = [axes]

    label_fs = 16
    tick_fs  = 13
    band_indices = np.arange(N_MELS)

    for ax, sp in zip(axes, all_species):
        sp_data = species_band_data[sp]
        cls_names = list(sp_data.keys())
        n_cls = len(cls_names)
        if n_cls == 0:
            ax.axis("off")
            continue

        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_cls)
        colors = [cmap(i) for i in range(n_cls)]

        for idx, (cls_name, data) in enumerate(sp_data.items()):
            # data: (N, 64) → 取各 mel 频带均值（代表每类整体能量轮廓）
            mean_profile = data.mean(axis=0)       # (64,)
            std_profile  = data.std(axis=0)        # (64,)
            ax.plot(band_indices, mean_profile,
                    color=colors[idx], linewidth=2.0,
                    label=cls_name, alpha=0.9)
            ax.fill_between(band_indices,
                            mean_profile - std_profile,
                            mean_profile + std_profile,
                            color=colors[idx], alpha=0.12)

        sp_cn = "狗" if "dog" in sp.lower() else "猫"
        ax.set_xlabel("梅尔频带索引 (0=低频, 63=高频)", fontproperties=fp, fontsize=label_fs)
        ax.set_ylabel("平均 log-mel 能量 (dB)", fontproperties=fp, fontsize=label_fs)
        ax.set_title(f"{sp_cn}音频各类别频带能量对比（均值 ± std）",
                     fontproperties=fp, fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.set_facecolor("white")
        ax.set_xlim(0, N_MELS - 1)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        legend = ax.legend(
            prop=fp, fontsize=tick_fs,
            loc="lower right", framealpha=0.9,
            edgecolor="#aaaaaa",
        )

    plt.tight_layout(pad=2.0)
    out = FIG_DIR / f"mel_spec_band_energy_{TS}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("已保存：%s", out)
    return out

# ============================================================
# 绘图 3：频带能量分布箱线图
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("梅尔频谱可视化调试工具启动")
    logger.info("CED-Mini 参数: SR=%d, n_fft=%d, hop=%d, n_mels=%d, top_db=%d",
                SR, N_FFT, HOP_SIZE, N_MELS, MEL_TOP_DB)
    logger.info("=" * 60)

    fp = _get_zh_font()

    # 扫描数据集
    dog_classes = scan_classes(DOG_AUDIO_DIR)
    cat_classes = scan_classes(CAT_AUDIO_DIR)
    logger.info("狗音频类别：%s（各类文件数：%s）",
                list(dog_classes.keys()),
                {k: len(v) for k, v in dog_classes.items()})
    logger.info("猫音频类别：%s（各类文件数：%s）",
                list(cat_classes.keys()),
                {k: len(v) for k, v in cat_classes.items()})

    # 构建梅尔变换器（复用，避免重复初始化）
    mel_spec, amp_to_db = build_mel_transform()

    all_class_files = {"dog": dog_classes, "cat": cat_classes}

    # 取目标类别文件列表
    dog_growling = dog_classes.get("growling", [])
    cat_angry    = cat_classes.get("Angry", [])
    if not dog_growling:
        logger.error("未找到狗 growling 目录，请检查路径")
    if not cat_angry:
        logger.error("未找到猫 Angry 目录，请检查路径")

    # ── 图1：狗 growling 1×4 梅尔频谱 ─────────────────────
    logger.info("\n[1/3] 绘制狗 growling 1×4 梅尔频谱图...")
    if dog_growling:
        plot_mel_1x4(dog_growling, mel_spec, amp_to_db, fp,
                     cls_name="growling", species_cn="狗", out_key="growling")

    # ── 图2：猫 Angry 1×4 梅尔频谱 ────────────────────────
    logger.info("\n[2/3] 绘制猫 Angry 1×4 梅尔频谱图...")
    if cat_angry:
        plot_mel_1x4(cat_angry, mel_spec, amp_to_db, fp,
                     cls_name="Angry", species_cn="猫", out_key="angry")

    # ── 图3：growling vs Angry 波形 + 梅尔频谱对比 ─────────
    logger.info("\n[3/3] 绘制 growling vs Angry 波形 + 梅尔频谱对比图...")
    if dog_growling and cat_angry:
        plot_waveform_vs_mel(dog_growling, cat_angry, mel_spec, amp_to_db, fp)

    # ── 图4：各类别频带能量轮廓对比 ────────────────────────
    logger.info("\n[额外] 绘制各类别频带能量轮廓对比图...")
    plot_band_energy_boxplot(all_class_files, mel_spec, amp_to_db, fp, n_per_class=N_BOX_SAMPLES)

    logger.info("\n全部可视化完成！")
    logger.info("日志保存至：%s", log_path)


if __name__ == "__main__":
    main()
