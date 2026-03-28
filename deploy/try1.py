# -*- coding: utf-8 -*-
"""
宠物多模态情绪识别 —— 决策级语义映射融合推理系统

融合策略设计依据：
  视觉与声学两模态的原始标签分属不同语义层次——
  图像标签（angry/happy/…）偏情绪外观，音频标签（barking/growling/…）偏发声行为。
  二者不可直接投票或拼接。本系统引入「统一状态空间」作为语义桥梁，
  将异构标签通过行为学先验软映射矩阵投射到同一维度后，
  再以置信度自适应加权融合，实现跨模态互补与消歧。

  犬类统一状态 (4 维)：威胁/攻击 · 积极/兴奋 · 放松/平静 · 低落/不安
  猫类统一状态 (5 维)：攻击/防御 · 恐惧/警戒 · 积极/亲和 · 平静/中性 · 低落/不适
  猫类额外保留特殊行为标签：狩猎专注 · 繁殖相关 · 母性呼叫 · 疼痛预警

行为学映射参考：
  - Russell (1980) 情感环形模型 (效价-唤醒度)
  - Beerda et al. (1997) 犬类行为应激指标
  - Turner & Bateson (2014) 家猫行为与福利
  - Mills et al. (2016) 犬类声学通信语义

运行：
  conda activate d2l
  python deploy/try1.py --image <图像路径> --audio <音频路径>
  python deploy/try1.py --image <图像路径>           # 仅视觉
  python deploy/try1.py --audio <音频路径>            # 仅声学
  python deploy/try1.py --demo                        # 自动选取样本演示

输入：宠物图像 (.jpg/.png/…) 和/或音频 (.wav/.mp3/…)
输出：结构化融合结果（物种/主状态/行为/置信度/一致性/解释）+ 终端日志 txt
"""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T_audio
from PIL import Image
from torchvision import transforms

# ════════════════════════════════════════════════════════════════
# 路径与全局常量
# ════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "moxing"
TXT_DIR = ROOT / "txt"
TXT_DIR.mkdir(parents=True, exist_ok=True)

DINOV3_PRETRAINED = str(
    ROOT / "pretrained_modelscope" / "facebook"
    / "dinov3-convnext-tiny-pretrain-lvd1689m"
)

TS = datetime.now().strftime("%Y%m%d%H%M%S")
_LOG_PATH = TXT_DIR / f"fusion_infer_{TS}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("fusion")

# ════════════════════════════════════════════════════════════════
# 第一部分 · 标签体系与行为学语义映射知识库
# ════════════════════════════════════════════════════════════════

DOG_IMG_CLASSES = ["angry", "happy", "relaxed", "sad"]
CAT_IMG_CLASSES = [
    "Angry", "Disgusted", "Happy", "Normal", "Sad", "Scared", "Surprised",
]
DOG_AUDIO_CLASSES = ["barking", "growling", "howling", "whining"]
CAT_AUDIO_CLASSES = [
    "Angry", "Defence", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning",
]

# --- 统一状态空间 ---
DOG_UNIFIED_STATES = ["威胁/攻击", "积极/兴奋", "放松/平静", "低落/不安"]
CAT_UNIFIED_STATES = ["攻击/防御", "恐惧/警戒", "积极/亲和", "平静/中性", "低落/不适"]

# --- 猫特殊行为标签 (音频类名 → (中文标签, 触发阈值)) ---
# Paining 阈值更低：疼痛/不适是高优先级异常信号，需更灵敏地捕获
CAT_SPECIAL_BEHAVIORS: dict[str, tuple[str, float]] = {
    "HuntingMind": ("狩猎专注", 0.30),
    "Mating":      ("繁殖相关", 0.30),
    "MotherCall":  ("母性呼叫", 0.30),
    "Paining":     ("疼痛预警", 0.25),
}
# Paining 高置信覆盖阈值：当音频对 Paining 的 softmax 概率超过此值时，
# 无论图像如何，最终主状态强制归入「低落/不适」
PAINING_OVERRIDE_THRESHOLD = 0.40

# --- 语义映射矩阵 (行为学先验) ---
# 每行对应一个原始类别，每列对应统一状态空间中的一个维度；行和为 1。
# 狗：图像→统一 (4×4)
M_IMG_DOG = np.array([
    #  威胁/攻击  积极/兴奋  放松/平静  低落/不安
    [  0.85,      0.00,      0.00,      0.15],   # angry
    [  0.00,      0.85,      0.10,      0.05],   # happy
    [  0.00,      0.05,      0.90,      0.05],   # relaxed
    [  0.05,      0.00,      0.05,      0.90],   # sad
], dtype=np.float64)

# 狗：音频→统一 (4×4)
# barking 高度分散——吠叫是犬类最多义的发声行为，必须依赖视觉消歧。
# howling 偏低落/不安但保留部分分散（嚎叫可能是社交呼唤而非纯悲伤）。
M_AUDIO_DOG = np.array([
    #  威胁/攻击  积极/兴奋  放松/平静  低落/不安
    [  0.30,      0.40,      0.05,      0.25],   # barking  (最模糊)
    [  0.85,      0.05,      0.00,      0.10],   # growling (高威胁确定性)
    [  0.15,      0.05,      0.10,      0.70],   # howling  (偏呼唤/孤立)
    [  0.05,      0.00,      0.10,      0.85],   # whining  (高求助确定性)
], dtype=np.float64)

# 猫：图像→统一 (7×5)
# Disgusted 在猫类行为学中更接近轻度不适/排斥，故主要映射至低落/不适。
# Surprised 主要映射至恐惧/警戒（猫的惊讶面容多为突发警觉反应）。
M_IMG_CAT = np.array([
    #  攻击/防御  恐惧/警戒  积极/亲和  平静/中性  低落/不适
    [  0.85,      0.10,      0.00,      0.00,      0.05],   # Angry
    [  0.20,      0.15,      0.00,      0.10,      0.55],   # Disgusted
    [  0.00,      0.00,      0.85,      0.10,      0.05],   # Happy
    [  0.00,      0.05,      0.10,      0.80,      0.05],   # Normal
    [  0.00,      0.05,      0.00,      0.05,      0.90],   # Sad
    [  0.05,      0.80,      0.00,      0.05,      0.10],   # Scared
    [  0.10,      0.60,      0.05,      0.10,      0.15],   # Surprised
], dtype=np.float64)

# 猫：音频→统一 (10×5)
# Defence 在攻击/防御与恐惧/警戒之间近似对半（防御态猫同时具有攻击性和恐惧）。
# HuntingMind 主要映射至平静/中性（狩猎专注状态外观常很平静，但唤醒度高）。
# Mating 分布分散（繁殖发声涉及不适、亲和、激动等多重维度）。
# MotherCall 主要映射至积极/亲和（母性社交呼叫）。
# Paining 强映射至低落/不适（痛苦是最高优先级的异常信号）。
M_AUDIO_CAT = np.array([
    #  攻击/防御  恐惧/警戒  积极/亲和  平静/中性  低落/不适
    [  0.85,      0.10,      0.00,      0.00,      0.05],   # Angry
    [  0.40,      0.45,      0.00,      0.00,      0.15],   # Defence
    [  0.85,      0.05,      0.00,      0.00,      0.10],   # Fighting
    [  0.00,      0.00,      0.85,      0.10,      0.05],   # Happy
    [  0.10,      0.25,      0.05,      0.50,      0.10],   # HuntingMind
    [  0.05,      0.10,      0.25,      0.25,      0.35],   # Mating
    [  0.00,      0.05,      0.55,      0.30,      0.10],   # MotherCall
    [  0.05,      0.10,      0.00,      0.00,      0.85],   # Paining
    [  0.00,      0.00,      0.10,      0.85,      0.05],   # Resting
    [  0.20,      0.60,      0.00,      0.05,      0.15],   # Warning
], dtype=np.float64)

# --- 描述性文本（用于自然语言解释生成）---
DOG_IMG_DESC = {
    "angry":   "面部紧绷、姿态警戒",
    "happy":   "表情愉悦、姿态积极",
    "relaxed": "表情舒缓、姿态放松",
    "sad":     "表情低沉、姿态萎靡",
}
DOG_AUDIO_DESC = {
    "barking":  "吠叫",
    "growling": "低吼/咆哮",
    "howling":  "嚎叫/长嗥",
    "whining":  "呜咽/哀鸣",
}
CAT_IMG_DESC = {
    "Angry":     "面部攻击性姿态",
    "Disgusted": "面部排斥/厌恶表情",
    "Happy":     "面部放松愉悦",
    "Normal":    "面部表情平和",
    "Sad":       "面部低沉忧郁",
    "Scared":    "面部惊恐警觉",
    "Surprised": "面部突然警觉",
}
CAT_AUDIO_DESC = {
    "Angry":      "愤怒嘶吼",
    "Defence":    "防御性发声",
    "Fighting":   "打斗嘶叫",
    "Happy":      "愉悦呼噜",
    "HuntingMind":"狩猎专注叫声",
    "Mating":     "求偶发声",
    "MotherCall": "母性呼唤",
    "Paining":    "痛苦哀鸣",
    "Resting":    "安静休息",
    "Warning":    "警告性发声",
}

# ════════════════════════════════════════════════════════════════
# 第二部分 · DINOv3 图像模型架构（精确复现训练代码 state_dict 结构）
# ════════════════════════════════════════════════════════════════

INPUT_SIZE = 224


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class SEGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 32)
        self.gate = nn.Sequential(
            nn.Linear(dim, mid), nn.GELU(),
            nn.Linear(mid, dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


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
        self.skip_proj = (nn.Linear(in_dim, mid) if in_dim != mid
                          else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(x)
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = F.gelu(self.ln2(self.fc2(h)))
        h = self.drop(h) + skip
        return self.classifier(h)


class DINOv3MultibranchModel(nn.Module):
    def __init__(self, model_name: str = DINOV3_PRETRAINED, hidden_dim: int = 384):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        all_dims = list(self.backbone.config.hidden_sizes)
        self.use_stage_indices = list(
            range(max(0, len(all_dims) - 2), len(all_dims))
        )
        self.stage_dims = [all_dims[i] for i in self.use_stage_indices]
        total_dim = sum(self.stage_dims)

        self.stage_gems = nn.ModuleList(
            [GeM(p=3.0) for _ in self.stage_dims]
        )
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.se_gate = SEGate(hidden_dim, reduction=4)
        self.head_species = nn.Linear(hidden_dim, 2)
        self.head_dog_img = ResidualTaskHead(
            hidden_dim, len(DOG_IMG_CLASSES), dropout=0.25,
        )
        self.head_cat_img = ResidualTaskHead(
            hidden_dim, len(CAT_IMG_CLASSES), dropout=0.35,
        )

    def _aggregate_features(self, hidden_states: tuple) -> torch.Tensor:
        all_stage_feats = hidden_states[1:]
        parts = []
        for idx, gem in zip(self.use_stage_indices, self.stage_gems):
            feat = all_stage_feats[idx]
            parts.append(gem(feat) if feat.dim() == 4 else feat.mean(dim=1))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        species: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        h = self.projection(self._aggregate_features(outputs.hidden_states))
        h = self.se_gate(h)

        species_logits = self.head_species(h)
        route = species if species is not None else species_logits.argmax(dim=1)

        dog_logits = h.new_zeros(h.shape[0], len(DOG_IMG_CLASSES))
        cat_logits = h.new_zeros(h.shape[0], len(CAT_IMG_CLASSES))
        dog_mask, cat_mask = (route == 0), (route == 1)
        if dog_mask.any():
            dog_logits[dog_mask] = self.head_dog_img(h[dog_mask]).to(dog_logits.dtype)
        if cat_mask.any():
            cat_logits[cat_mask] = self.head_cat_img(h[cat_mask]).to(cat_logits.dtype)

        return {"species": species_logits, "dog_img": dog_logits, "cat_img": cat_logits}


# ════════════════════════════════════════════════════════════════
# 第三部分 · 预处理
# ════════════════════════════════════════════════════════════════

class PadToSquare:
    def __init__(self, fill: int = 128):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        diff = abs(w - h)
        pad1, pad2 = diff // 2, diff - diff // 2
        padding = (0, pad1, 0, pad2) if w > h else (pad1, 0, pad2, 0)
        return transforms.functional.pad(
            img, padding, fill=self.fill, padding_mode="constant",
        )


def _build_image_transform(mean: list[float], std: list[float]):
    return transforms.Compose([
        PadToSquare(fill=128),
        transforms.Resize(
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_image(path: str | Path, transform) -> torch.Tensor:
    """加载单张图像并预处理为 (1, 3, H, W) 张量。"""
    with Image.open(path) as img:
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (128, 128, 128))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
        tensor = transform(img)
    return tensor.unsqueeze(0)


def load_audio_mel(path: str | Path) -> torch.Tensor:
    """
    加载单条音频并转换为 CED-Mini 所需的 (1, n_mels, T) log-mel 频谱。
    参数与 cedtrain/config.py TrainConfig 默认值严格对齐。
    """
    SR = 16000
    MAX_SAMPLES = int(10.0 * SR)
    CLIP_SAMPLES = int(10.0 * SR)

    waveform, sr = torchaudio.load(str(path))
    if sr != SR:
        waveform = T_audio.Resample(sr, SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    t_len = waveform.shape[1]
    if t_len == 0:
        raise ValueError(f"空音频: {path}")
    if t_len > MAX_SAMPLES:
        waveform = waveform[:, :MAX_SAMPLES]
        t_len = MAX_SAMPLES
    waveform = waveform / waveform.abs().max().clamp(min=1e-8)
    if t_len < CLIP_SAMPLES:
        reps = (CLIP_SAMPLES + t_len - 1) // t_len
        waveform = torch.tile(waveform, (1, reps))[:, :CLIP_SAMPLES]

    mel_spec = T_audio.MelSpectrogram(
        sample_rate=SR, n_fft=512, win_length=512, hop_length=160,
        f_min=0, f_max=8000, n_mels=64, center=True,
    )
    amp_to_db = T_audio.AmplitudeToDB(top_db=120)
    mel = amp_to_db(mel_spec(waveform)).squeeze(0)  # (n_mels, T)
    return mel.unsqueeze(0)  # (1, n_mels, T)


# ════════════════════════════════════════════════════════════════
# 第四部分 · 模型加载
# ════════════════════════════════════════════════════════════════

def _find_latest(pattern: str) -> Path:
    candidates = sorted(MODEL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"在 {MODEL_DIR} 下未找到匹配 '{pattern}' 的模型文件"
        )
    return candidates[-1]


def load_image_model(
    device: torch.device,
    ckpt_path: Path | None = None,
) -> tuple[DINOv3MultibranchModel, transforms.Compose]:
    if ckpt_path is None:
        ckpt_path = _find_latest("DINOv3_ConvNeXt_*.pkl")
    logger.info(f"加载图像模型: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_states"][0]

    model = DINOv3MultibranchModel(model_name=DINOV3_PRETRAINED, hidden_dim=384)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(DINOV3_PRETRAINED)
    tfm = _build_image_transform(proc.image_mean, proc.image_std)

    logger.info(f"  图像模型就绪 (设备={device})")
    return model, tfm


def load_audio_model(
    device: torch.device,
    ckpt_path: Path | None = None,
):
    if ckpt_path is None:
        ckpt_path = _find_latest("CedMini_AudioEmotion_*.pkl")
    logger.info(f"加载音频模型: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_states"][0]

    from cedtrain.config import TrainConfig as AudioCfg
    from cedtrain.modeling import CedAudioEmotionModel

    model = CedAudioEmotionModel(AudioCfg())
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    logger.info(f"  音频模型就绪 (设备={device})")
    return model


# ════════════════════════════════════════════════════════════════
# 第五部分 · 融合结果数据结构
# ════════════════════════════════════════════════════════════════

@dataclass
class FusionResult:
    species: str
    species_confidence: float
    primary_state: str
    primary_confidence: float
    unified_distribution: dict[str, float]
    image_prediction: str | None
    image_confidence: float | None
    audio_prediction: str | None
    audio_confidence: float | None
    special_behavior: str | None
    risk_level: str
    consistency: str
    consistency_score: float | None
    explanation: str
    modality_weights: dict[str, float]


# ════════════════════════════════════════════════════════════════
# 第六部分 · 融合引擎
# ════════════════════════════════════════════════════════════════

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _normalized_entropy(probs: np.ndarray) -> float:
    """归一化信息熵 ∈ [0,1]。0 = 完全确定；1 = 完全均匀。"""
    eps = 1e-12
    h = -np.sum(probs * np.log(probs + eps))
    return float(h / np.log(len(probs)))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _consistency_label(score: float) -> str:
    if score >= 0.85:
        return "高度一致"
    if score >= 0.60:
        return "基本一致"
    if score >= 0.35:
        return "存在分歧"
    return "模态冲突"


def _fuse_species(
    sp_img: np.ndarray | None,
    sp_audio: np.ndarray | None,
) -> tuple[int, float]:
    """融合物种预测。返回 (species_id, confidence)。"""
    if sp_img is not None and sp_audio is not None:
        fused = 0.5 * sp_img + 0.5 * sp_audio
    elif sp_img is not None:
        fused = sp_img
    else:
        fused = sp_audio
    idx = int(np.argmax(fused))
    return idx, float(fused[idx])


def _determine_risk(
    species: str,
    state: str,
    conf: float,
    special: str | None,
) -> str:
    high_risk_states = {"威胁/攻击", "攻击/防御"}
    monitor_states = {"低落/不安", "低落/不适", "恐惧/警戒"}

    if special == "疼痛预警":
        return "警惕"
    if state in high_risk_states and conf > 0.45:
        return "警惕"
    if state in monitor_states and conf > 0.40:
        return "关注"
    return "正常"


def fuse(
    img_logits: dict[str, np.ndarray] | None,
    audio_logits: dict[str, np.ndarray] | None,
    species_hint: str | None = None,
) -> FusionResult:
    """
    核心融合函数。

    img_logits:   {"species": (2,), "dog_img": (4,), "cat_img": (7,)} 或 None
    audio_logits: {"species": (2,), "dog_audio": (4,), "cat_audio": (10,)} 或 None
    """
    if img_logits is None and audio_logits is None:
        raise ValueError("至少需要提供一个模态的推理结果")

    # ── 1. 物种判定 ──
    sp_img = _softmax(img_logits["species"]) if img_logits is not None else None
    sp_audio = _softmax(audio_logits["species"]) if audio_logits is not None else None

    if species_hint is not None:
        species_id = 0 if species_hint.lower() in ("dog", "狗") else 1
        species_conf = 1.0
    else:
        species_id, species_conf = _fuse_species(sp_img, sp_audio)

    species_name = "狗" if species_id == 0 else "猫"
    is_dog = (species_id == 0)

    # ── 2. 提取对应分支的 softmax 概率 ──
    p_img, p_audio = None, None
    img_classes = DOG_IMG_CLASSES if is_dog else CAT_IMG_CLASSES
    audio_classes = DOG_AUDIO_CLASSES if is_dog else CAT_AUDIO_CLASSES
    img_key = "dog_img" if is_dog else "cat_img"
    audio_key = "dog_audio" if is_dog else "cat_audio"
    M_img = M_IMG_DOG if is_dog else M_IMG_CAT
    M_audio = M_AUDIO_DOG if is_dog else M_AUDIO_CAT
    unified_names = DOG_UNIFIED_STATES if is_dog else CAT_UNIFIED_STATES
    img_desc_map = DOG_IMG_DESC if is_dog else CAT_IMG_DESC
    audio_desc_map = DOG_AUDIO_DESC if is_dog else CAT_AUDIO_DESC

    if img_logits is not None:
        p_img = _softmax(img_logits[img_key])
    if audio_logits is not None:
        p_audio = _softmax(audio_logits[audio_key])

    img_pred, img_conf = None, None
    audio_pred, audio_conf = None, None
    if p_img is not None:
        img_pred = img_classes[int(np.argmax(p_img))]
        img_conf = float(np.max(p_img))
    if p_audio is not None:
        audio_pred = audio_classes[int(np.argmax(p_audio))]
        audio_conf = float(np.max(p_audio))

    # ── 3. 映射到统一状态空间 ──
    #    p_unified = M^T @ p_original  (矩阵行=原始类, 列=统一维度)
    p_unified_img = M_img.T @ p_img if p_img is not None else None
    p_unified_audio = M_audio.T @ p_audio if p_audio is not None else None

    # ── 4. 动态加权 ──
    BASE_W_IMG, BASE_W_AUDIO = 0.60, 0.40

    if p_unified_img is not None and p_unified_audio is not None:
        cert_img = 1.0 - _normalized_entropy(p_img)
        cert_audio = 1.0 - _normalized_entropy(p_audio)
        w_img = BASE_W_IMG * max(cert_img, 0.15)
        w_audio = BASE_W_AUDIO * max(cert_audio, 0.15)

        # 犬类 barking 歧义惩罚：barking 为 top-1 时降低音频权重
        if is_dog and audio_pred == "barking":
            w_audio *= 0.85

        total_w = w_img + w_audio
        w_img /= total_w
        w_audio /= total_w

        p_final = w_img * p_unified_img + w_audio * p_unified_audio
        consistency_score = _cosine_sim(p_unified_img, p_unified_audio)
        consistency = _consistency_label(consistency_score)
    elif p_unified_img is not None:
        p_final = p_unified_img
        w_img, w_audio = 1.0, 0.0
        consistency, consistency_score = "仅视觉模态", None
    else:
        p_final = p_unified_audio
        w_img, w_audio = 0.0, 1.0
        consistency, consistency_score = "仅声学模态", None

    # ── 5. 猫 Paining 覆盖逻辑 ──
    special_behavior = None
    if not is_dog and p_audio is not None:
        paining_idx = CAT_AUDIO_CLASSES.index("Paining")
        if p_audio[paining_idx] >= PAINING_OVERRIDE_THRESHOLD:
            discomfort_idx = CAT_UNIFIED_STATES.index("低落/不适")
            p_final = np.zeros_like(p_final)
            p_final[discomfort_idx] = 1.0
            special_behavior = "疼痛预警"
        else:
            for cls_name, (tag_cn, threshold) in CAT_SPECIAL_BEHAVIORS.items():
                cls_idx = CAT_AUDIO_CLASSES.index(cls_name)
                if p_audio[cls_idx] >= threshold:
                    if special_behavior is None or cls_name == "Paining":
                        special_behavior = tag_cn

    # ── 6. 输出 ──
    p_final = p_final / (p_final.sum() + 1e-12)
    primary_idx = int(np.argmax(p_final))
    primary_state = unified_names[primary_idx]
    primary_conf = float(p_final[primary_idx])

    risk = _determine_risk(species_name, primary_state, primary_conf, special_behavior)

    unified_dist = {name: float(p_final[i]) for i, name in enumerate(unified_names)}

    explanation = _generate_explanation(
        species_name, primary_state, primary_conf,
        img_pred, img_conf, img_desc_map,
        audio_pred, audio_conf, audio_desc_map,
        special_behavior, consistency, risk,
    )

    return FusionResult(
        species=species_name,
        species_confidence=species_conf,
        primary_state=primary_state,
        primary_confidence=primary_conf,
        unified_distribution=unified_dist,
        image_prediction=img_pred,
        image_confidence=img_conf,
        audio_prediction=audio_pred,
        audio_confidence=audio_conf,
        special_behavior=special_behavior,
        risk_level=risk,
        consistency=consistency,
        consistency_score=consistency_score,
        explanation=explanation,
        modality_weights={"图像": round(w_img, 4), "音频": round(w_audio, 4)},
    )


# ════════════════════════════════════════════════════════════════
# 第七部分 · 解释生成
# ════════════════════════════════════════════════════════════════

def _generate_explanation(
    species: str,
    state: str,
    conf: float,
    img_pred: str | None,
    img_conf: float | None,
    img_desc: dict,
    audio_pred: str | None,
    audio_conf: float | None,
    audio_desc: dict,
    special: str | None,
    consistency: str,
    risk: str,
) -> str:
    parts: list[str] = []

    if img_pred is not None and audio_pred is not None:
        img_d = img_desc.get(img_pred, img_pred)
        audio_d = audio_desc.get(audio_pred, audio_pred)

        if "一致" in consistency:
            parts.append(
                f"视觉与声学模态{consistency}：图像显示{img_d}"
                f"（{img_pred} {img_conf:.0%}），"
                f"音频检测到{audio_d}（{audio_pred} {audio_conf:.0%}），"
                f"综合判定处于「{state}」状态。"
            )
        elif consistency == "存在分歧":
            parts.append(
                f"视觉与声学模态存在分歧：图像显示{img_d}"
                f"（{img_pred} {img_conf:.0%}），"
                f"但音频为{audio_d}（{audio_pred} {audio_conf:.0%}），"
                f"融合后倾向「{state}」，但建议结合实际场景判断。"
            )
        else:
            parts.append(
                f"视觉与声学模态冲突：图像显示{img_d}，"
                f"而音频为{audio_d}，二者语义不一致，"
                f"融合结果「{state}」置信度偏低（{conf:.0%}），"
                f"建议谨慎解读。"
            )
    elif img_pred is not None:
        img_d = img_desc.get(img_pred, img_pred)
        parts.append(
            f"仅视觉模态可用：图像显示{img_d}"
            f"（{img_pred} {img_conf:.0%}），判定为「{state}」。"
        )
    else:
        audio_d = audio_desc.get(audio_pred, audio_pred)
        parts.append(
            f"仅声学模态可用：音频检测到{audio_d}"
            f"（{audio_pred} {audio_conf:.0%}），判定为「{state}」。"
        )

    if special is not None:
        parts.append(f"音频同时触发特殊行为标签「{special}」，需额外关注。")

    if risk == "警惕":
        parts.append("风险等级为「警惕」，建议密切观察宠物状态。")
    elif risk == "关注":
        parts.append("风险等级为「关注」，建议适当留意。")

    return "".join(parts)


# ════════════════════════════════════════════════════════════════
# 第八部分 · 主推理系统
# ════════════════════════════════════════════════════════════════

class MultimodalPetEmotionSystem:
    """
    封装双模态加载与推理的高层接口。
    初始化时加载模型，之后可多次调用 infer()。
    """

    def __init__(
        self,
        device: str = "auto",
        img_ckpt: str | Path | None = None,
        audio_ckpt: str | Path | None = None,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass

        logger.info(f"计算设备: {self.device}")

        self.img_model, self.img_transform = load_image_model(
            self.device,
            Path(img_ckpt) if img_ckpt else None,
        )
        self.audio_model = load_audio_model(
            self.device,
            Path(audio_ckpt) if audio_ckpt else None,
        )

    @torch.inference_mode()
    def _infer_image(self, image_path: str | Path) -> dict[str, np.ndarray]:
        pixel_values = load_image(image_path, self.img_transform).to(self.device)
        out = self.img_model(pixel_values, species=None)
        return {k: v[0].cpu().float().numpy() for k, v in out.items()}

    @torch.inference_mode()
    def _infer_audio(self, audio_path: str | Path) -> dict[str, np.ndarray]:
        mel = load_audio_mel(audio_path).to(self.device)
        out = self.audio_model(mel, species=None)
        return {k: v[0].cpu().float().numpy() for k, v in out.items()}

    def _infer_image_forced(
        self, image_path: str | Path, species_id: int,
    ) -> dict[str, np.ndarray]:
        """用强制物种路由重跑图像模型，确保目标分支 logits 有效。"""
        pixel_values = load_image(image_path, self.img_transform).to(self.device)
        sp = torch.tensor([species_id], device=self.device)
        with torch.inference_mode():
            out = self.img_model(pixel_values, species=sp)
        return {k: v[0].cpu().float().numpy() for k, v in out.items()}

    def _infer_audio_forced(
        self, audio_path: str | Path, species_id: int,
    ) -> dict[str, np.ndarray]:
        mel = load_audio_mel(audio_path).to(self.device)
        sp = torch.tensor([species_id], device=self.device)
        with torch.inference_mode():
            out = self.audio_model(mel, species=sp)
        return {k: v[0].cpu().float().numpy() for k, v in out.items()}

    def infer(
        self,
        image_path: str | Path | None = None,
        audio_path: str | Path | None = None,
        species_hint: str | None = None,
    ) -> FusionResult:
        if image_path is None and audio_path is None:
            raise ValueError("至少需要提供 --image 或 --audio 之一")

        img_logits, audio_logits = None, None

        if image_path is not None:
            logger.info(f"图像推理: {image_path}")
            img_logits = self._infer_image(image_path)

        if audio_path is not None:
            logger.info(f"音频推理: {audio_path}")
            audio_logits = self._infer_audio(audio_path)

        # 确定融合物种并检查是否需要用强制路由重跑
        if species_hint is not None:
            final_sp = 0 if species_hint.lower() in ("dog", "狗") else 1
        else:
            sp_i = _softmax(img_logits["species"]) if img_logits else None
            sp_a = _softmax(audio_logits["species"]) if audio_logits else None
            final_sp, _ = _fuse_species(sp_i, sp_a)

        if img_logits is not None:
            img_route = int(np.argmax(img_logits["species"]))
            if img_route != final_sp:
                logger.info(f"  图像模型内部路由({img_route})与融合物种({final_sp})不一致，重跑强制路由")
                img_logits = self._infer_image_forced(image_path, final_sp)

        if audio_logits is not None:
            audio_route = int(np.argmax(audio_logits["species"]))
            if audio_route != final_sp:
                logger.info(f"  音频模型内部路由({audio_route})与融合物种({final_sp})不一致，重跑强制路由")
                audio_logits = self._infer_audio_forced(audio_path, final_sp)

        result = fuse(img_logits, audio_logits, species_hint)
        return result


# ════════════════════════════════════════════════════════════════
# 第九部分 · 结果展示
# ════════════════════════════════════════════════════════════════

def format_result(r: FusionResult) -> str:
    W = 62
    lines: list[str] = []
    lines.append("═" * W)
    lines.append("  宠物多模态情绪识别结果")
    lines.append("═" * W)

    lines.append(f"  物种判定：  {r.species} (置信度 {r.species_confidence:.1%})")
    lines.append(f"  主情绪状态：{r.primary_state} (置信度 {r.primary_confidence:.1%})")

    if r.audio_prediction is not None:
        audio_desc_all = {**DOG_AUDIO_DESC, **CAT_AUDIO_DESC}
        audio_cn = audio_desc_all.get(r.audio_prediction, "")
        lines.append(f"  声音行为：  {r.audio_prediction} ({audio_cn})")
    if r.special_behavior is not None:
        lines.append(f"  特殊行为：  {r.special_behavior}")
    lines.append(f"  风险等级：  {r.risk_level}")
    if r.consistency_score is not None:
        lines.append(f"  模态一致性：{r.consistency} ({r.consistency_score:.2f})")
    else:
        lines.append(f"  模态一致性：{r.consistency}")

    lines.append("─" * W)
    lines.append("  统一状态分布：")
    max_key = max(r.unified_distribution, key=r.unified_distribution.get)
    for name, val in r.unified_distribution.items():
        bar_len = int(val * 30)
        bar = "█" * bar_len
        marker = " << 最高" if name == max_key else ""
        lines.append(f"    {name:<8s}: {val:>6.1%}  {bar}{marker}")

    lines.append("─" * W)
    if r.image_prediction is not None:
        lines.append(f"  图像证据：  {r.image_prediction} ({r.image_confidence:.1%})")
    if r.audio_prediction is not None:
        lines.append(f"  音频证据：  {r.audio_prediction} ({r.audio_confidence:.1%})")
    lines.append(
        f"  融合权重：  图像 {r.modality_weights['图像']:.1%}"
        f" / 音频 {r.modality_weights['音频']:.1%}"
    )

    lines.append("─" * W)
    lines.append("  综合解释：")
    exp = r.explanation
    chunk = W - 4
    for i in range(0, len(exp), chunk):
        lines.append(f"    {exp[i:i+chunk]}")

    lines.append("═" * W)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 第十部分 · 命令行入口
# ════════════════════════════════════════════════════════════════

def _find_demo_sample(data_dir: Path, extensions: set[str]) -> Path | None:
    """在目录树中找到第一个匹配的文件作为演示样本。"""
    if not data_dir.exists():
        return None
    for p in data_dir.rglob("*"):
        if p.suffix.lower() in extensions:
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="宠物多模态情绪识别 · 决策级语义映射融合推理",
    )
    parser.add_argument("--image", type=str, default=None, help="图像文件路径")
    parser.add_argument("--audio", type=str, default=None, help="音频文件路径")
    parser.add_argument(
        "--species", type=str, default=None, choices=["dog", "cat", "狗", "猫"],
        help="手动指定物种（跳过自动检测）",
    )
    parser.add_argument("--demo", action="store_true", help="自动选取样本演示")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--img-model", type=str, default=None, help="图像模型路径")
    parser.add_argument("--audio-model", type=str, default=None, help="音频模型路径")
    args = parser.parse_args()

    if args.demo:
        img_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        audio_ext = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

        img_candidates = [
            ROOT / "data" / "dog_emotion_cropped",
            ROOT / "data" / "cat_671_cropped",
        ]
        audio_candidates = [
            ROOT / "data" / "Pet dog sound event",
            ROOT / "data" / "CatSound_DataSet_V2" / "NAYA_DATA_AUG1X",
        ]

        for d in img_candidates:
            sample = _find_demo_sample(d, img_ext)
            if sample is not None:
                args.image = str(sample)
                break
        for d in audio_candidates:
            sample = _find_demo_sample(d, audio_ext)
            if sample is not None:
                args.audio = str(sample)
                break

        if args.image is None and args.audio is None:
            logger.error("未找到可用的演示样本文件")
            return
        logger.info(f"演示模式 - 图像: {args.image}")
        logger.info(f"演示模式 - 音频: {args.audio}")

    if args.image is None and args.audio is None:
        parser.print_help()
        return

    system = MultimodalPetEmotionSystem(
        device=args.device,
        img_ckpt=args.img_model,
        audio_ckpt=args.audio_model,
    )

    result = system.infer(
        image_path=args.image,
        audio_path=args.audio,
        species_hint=args.species,
    )

    output = format_result(result)
    print("\n" + output + "\n")
    logger.info(f"推理日志已保存: {_LOG_PATH}")


if __name__ == "__main__":
    main()
