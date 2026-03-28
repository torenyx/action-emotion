# -*- coding: utf-8 -*-
"""
大创项目：ImageBind 离线特征提取脚本（工业级）

功能：
  - 自动下载/加载 ImageBind Huge 预训练权重
  - 遍历四类数据（猫图/狗图/猫音频/狗音频），批量提取 1024 维特征
  - 将特征与标签保存为 .npy 文件，供后续 MLP 训练极速复用
  - 支持 fp16 半精度推理，RTX 4060 (8GB) 可完整运行

运行方式：
  conda activate d2l
  python train/extract_features.py

输出：
  data/features_npy/
    dog_img_feat.npy   (N, 1024)
    dog_img_label.npy  (N,)
    cat_img_feat.npy   (M, 1024)
    cat_img_label.npy  (M,)
    dog_audio_feat.npy (P, 1024)
    dog_audio_label.npy(P,)
    cat_audio_feat.npy (Q, 1024)
    cat_audio_label.npy(Q,)
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as audio_transforms
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# 路径配置
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "data" / "features_npy"
TXT_DIR = ROOT / "txt"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d%H%M%S")
log_path = TXT_DIR / f"extract_features_{TS}.txt"

class WarningFilter(logging.Filter):
    def filter(self, record):
        if "Large gap between audio n_frames" in record.getMessage():
            return False
        return True

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

# 为所有的 handler 添加过滤器，屏蔽刷屏的音频长度警告
for handler in logging.root.handlers:
    handler.addFilter(WarningFilter())

# ============================================================
# 数据路径与标签定义
# ============================================================
DOG_IMG_DIR    = ROOT / "data" / "dog_emotion_cropped"
CAT_IMG_DIR    = ROOT / "data" / "cat_671_cropped"
DOG_AUDIO_DIR  = ROOT / "data" / "Pet dog sound event"
CAT_AUDIO_DIR  = ROOT / "data" / "CatSound_DataSet_V2" / "NAYA_DATA_AUG1X"

DOG_IMG_CLASSES   = ["angry", "happy", "relaxed", "sad"]
CAT_IMG_CLASSES   = ["Angry", "Disgusted", "Happy", "Normal", "Sad", "Scared", "Surprised"]
DOG_AUDIO_CLASSES = ["barking", "growling", "howling", "whining"]
CAT_AUDIO_CLASSES = ["Angry", "Defence", "Fighting", "Happy", "HuntingMind",
                     "Mating", "MotherCall", "Paining", "Resting", "Warning"]

IMG_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# ImageBind 官方音频：Kaldi fbank → (1, 128, 204)，再按预训练统计量归一化
# 见 site-packages/imagebind/data.py 中 waveform2melspec
IMAGEBIND_SR         = 16000
IMAGEBIND_NUM_MELS   = 128
IMAGEBIND_TARGET_LEN = 204
IMAGEBIND_MEL_MEAN   = -4.268
IMAGEBIND_MEL_STD    = 9.138
# kaldi fbank 帧数公式：n_frames = floor((T - frame_length_samples) / hop_length) + 1
# frame_length=25ms → 400 采样点；frame_shift=10ms → hop_length=160 采样点
# 要精确得到 204 帧：T = (204-1)*160 + 400 = 32880 采样点（实测验证：32640→202帧，32880→204帧）
IMAGEBIND_HOP_LENGTH   = 160                                    # 10ms × 16000Hz
IMAGEBIND_FRAME_LENGTH = 400                                    # 25ms × 16000Hz
IMAGEBIND_CLIP_SAMPLES = (IMAGEBIND_TARGET_LEN - 1) * IMAGEBIND_HOP_LENGTH + IMAGEBIND_FRAME_LENGTH  # = 32880
# 超长音频在读取时截断，避免 torchaudio.load 撑爆内存（单文件上限 120 秒）
MAX_AUDIO_SEC     = 120
MAX_AUDIO_SAMPLES = MAX_AUDIO_SEC * IMAGEBIND_SR               # = 1_920_000 采样点


def waveform_to_imagebind_mel(waveform_mono_16k: torch.Tensor) -> torch.Tensor:
    """
    将 (1, T) 单声道 16kHz 波形转为 ImageBind 期望的 log-mel 张量 (1, 128, 204)，
    与 Meta 官方推理完全一致（kaldi fbank + Normalize）。
    """
    from imagebind.data import waveform2melspec

    mel = waveform2melspec(waveform_mono_16k.clone(), IMAGEBIND_SR,
                           IMAGEBIND_NUM_MELS, IMAGEBIND_TARGET_LEN)
    return transforms.Normalize(mean=IMAGEBIND_MEL_MEAN, std=IMAGEBIND_MEL_STD)(mel)

BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ImageBind 加载工具
# ============================================================
def load_imagebind():
    """
    加载 ImageBind Huge 模型。
    优先从 HuggingFace 镜像下载（国内可用），回退到官方源。
    """
    try:
        from imagebind import data as imagebind_data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        logger.info("检测到本地 imagebind 包，正在加载模型...")
    except ImportError:
        logger.error(
            "未安装 imagebind！请执行：\n"
            "  pip install git+https://github.com/facebookresearch/ImageBind.git\n"
            "或从 HuggingFace 镜像安装：\n"
            "  pip install git+https://hf-mirror.com/facebookresearch/ImageBind.git"
        )
        sys.exit(1)

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    # fp16 推理节省显存（RTX 4060 8GB 完全足够）
    if DEVICE.type == "cuda":
        model = model.half().to(DEVICE)
    else:
        model = model.to(DEVICE)

    logger.info(f"ImageBind Huge 加载完成，运行在 {DEVICE}（{'fp16' if DEVICE.type == 'cuda' else 'fp32'}）")
    return model, imagebind_data, ModalityType


# ============================================================
# 图像 Dataset
# ============================================================
class PadToSquare:
    """
    将图像填充为正方形，避免直接 Resize 导致的形变，
    或 CenterCrop 导致的边缘特征（如耳朵、下巴）丢失。
    这对已经裁切好（_cropped）的面部/主体数据集至关重要。
    """
    def __init__(self, fill=128):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        # padding = (left, top, right, bottom)
        padding = (0, pad1, 0, pad2) if w > h else (pad1, 0, pad2, 0)
        return transforms.functional.pad(img, padding, fill=self.fill, padding_mode='constant')


class ImageFolderDataset(Dataset):
    """从 cls_dir/class_name/*.jpg 结构扫描图像"""

    # 针对已裁切数据集优化的预处理：
    # 填充至正方形 -> 缩放至 224 -> ToTensor -> Normalize
    _TRANSFORM = transforms.Compose([
        PadToSquare(fill=128),  # 使用灰色(128)填充空白区域，更接近自然中性色
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    def __init__(self, root: Path, class_list: list[str]):
        self.samples: list[tuple[Path, int]] = []
        skipped = 0
        for label, cls in enumerate(class_list):
            cls_dir = root / cls
            if not cls_dir.exists():
                logger.warning(f"类别目录不存在，已跳过：{cls_dir}")
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                # 预检：提前过滤无法打开的损坏文件，避免全零占位污染特征
                try:
                    with Image.open(p) as im:
                        im.verify()
                    self.samples.append((p, label))
                except Exception as e:
                    logger.warning(f"损坏图像已剔除 {p}: {e}")
                    skipped += 1
        logger.info(f"  共扫描到 {len(self.samples)} 张图像，来自 {root.name}"
                    + (f"（剔除损坏 {skipped} 张）" if skipped else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                # RGBA（带透明通道的 PNG）：先贴到同色灰底上再转 RGB，
                # 避免 convert("RGB") 直接将透明区域填成纯黑而影响特征
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (128, 128, 128))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")
                img = self._TRANSFORM(img)
            return img, label
        except Exception as e:
            logger.warning(f"读取图像数据失败已动态剔除 {path}: {e}")
            return None


def collate_drop_none(batch):
    """过滤掉 __getitem__ 返回 None 的损坏样本"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    return imgs, labels


# ============================================================
# 音频 Dataset
# ============================================================
class AudioFolderDataset(Dataset):
    """
    从 cls_dir/class_name/*.wav 结构扫描音频，仅收集路径与标签。
    真正的预处理（Loop Padding / 均匀切段 / Mel 转换）在
    extract_audio_features 中逐条完成，以支持任意时长的音频文件。
    """

    def __init__(self, root: Path, class_list: list[str]):
        self.samples: list[tuple[Path, int]] = []
        skipped = 0
        for label, cls in enumerate(class_list):
            cls_dir = root / cls
            if not cls_dir.exists():
                logger.warning(f"类别目录不存在，已跳过：{cls_dir}")
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                # 预检：过滤无法解析的损坏音频
                try:
                    torchaudio.info(str(p))
                    self.samples.append((p, label))
                except Exception as e:
                    logger.warning(f"损坏音频已剔除 {p}: {e}")
                    skipped += 1
        logger.info(f"  共扫描到 {len(self.samples)} 条音频，来自 {root.name}"
                    + (f"（剔除损坏 {skipped} 条）" if skipped else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return str(path), label


# ============================================================
# 通用特征提取函数
# ============================================================
@torch.no_grad()
def extract_image_features(
    model, dataset: ImageFolderDataset, desc: str,
    multi_view: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    批量提取图像特征，返回 (features: (N,1024), labels: (N,))。

    multi_view=True 时，对每张图提取原图 + 水平翻转两个视角的特征并取均值，
    产出更鲁棒的、对左右方向不变的表征（学术论文标配做法，尤其对小样本数据集有效）。
    """
    from imagebind.models.imagebind_model import ModalityType

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
                        collate_fn=collate_drop_none)
    all_feats, all_labels = [], []

    for batch in tqdm(loader, desc=desc, ncols=80):
        if batch[0] is None:
            continue
        imgs, labels = batch

        if DEVICE.type == "cuda":
            imgs = imgs.half().to(DEVICE)
        else:
            imgs = imgs.to(DEVICE)

        # 原始视角
        emb_orig = model({ModalityType.VISION: imgs})[ModalityType.VISION]

        if multi_view:
            # 水平翻转视角
            imgs_flip = torch.flip(imgs, dims=[-1])
            emb_flip = model({ModalityType.VISION: imgs_flip})[ModalityType.VISION]
            # 两视角均值 → 更鲁棒的特征
            feats = ((emb_orig + emb_flip) / 2.0).cpu().float()
        else:
            feats = emb_orig.cpu().float()

        # 统一 L2 归一化到单位超球面
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1).numpy()
        all_feats.append(feats)
        all_labels.append(labels.numpy())

    return np.concatenate(all_feats), np.concatenate(all_labels)


def _load_waveform(path: str, clip_samples: int) -> torch.Tensor:
    """
    读取单条音频，重采样到 16kHz，混音到单声道，
    然后根据时长分两种情况处理：

      - 短于 clip_samples（< 2.055s）：
          Loop Padding——将波形首尾相连循环拼接到 >= clip_samples，
          再截取前 clip_samples 个采样点。
          保证频谱图每一帧都有真实声音能量，没有补零灰色区域。

      - 长于或等于 clip_samples（>= 2.055s）：
          保留完整波形（上限 MAX_AUDIO_SAMPLES 采样点，防止超长文件 OOM），
          由下游 _build_mel_clips 均匀切出 clips_per_audio 段，
          覆盖音频头、中、尾等不同位置，充分利用长音频信息。

    返回 (1, T') 单声道波形张量，clip_samples <= T' <= MAX_AUDIO_SAMPLES。

    注：猫音频数据集（CatSound_DataSet_V2）中约 10% 的 MP3 文件因录制/编码时
    响度归一化不当，解码后 PCM 振幅超出 [-1.0, 1.0]（最高达 2.07），
    直接送入 Kaldi fbank 会导致 log-mel 能量偏高，偏离 ImageBind 预训练分布。
    因此在所有预处理操作完成后统一做峰值归一化（peak normalization），
    将最大振幅缩放到 1.0 以内，消除响度偏差同时保留音色结构。
    """
    waveform, sr = torchaudio.load(path)
    if sr != IMAGEBIND_SR:
        waveform = audio_transforms.Resample(sr, IMAGEBIND_SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    T = waveform.shape[1]
    if T == 0:
        raise ValueError("Audio is empty (0 samples)")

    # 超长音频截断保护，避免读入数分钟录音撑爆内存
    if T > MAX_AUDIO_SAMPLES:
        waveform = waveform[:, :MAX_AUDIO_SAMPLES]
        T = MAX_AUDIO_SAMPLES

    # 峰值归一化：将最大振幅缩放至 1.0，消除 MP3 过载编码导致的响度偏差。
    # 使用 clamp 避免极静音文件（全零）除以零。
    peak = waveform.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        waveform = waveform / peak

    if T < clip_samples:
        # Loop Padding：循环拼接到恰好 clip_samples 个采样点
        repeats = (clip_samples + T - 1) // T
        waveform = torch.tile(waveform, (1, repeats))
        waveform = waveform[:, :clip_samples]

    # 长音频：保留完整波形，交给 _build_mel_clips 做均匀切段
    return waveform


def _build_mel_clips(waveform: torch.Tensor, clip_samples: int,
                     clips_per_audio: int = 5) -> torch.Tensor:
    """
    将完整波形切成若干段，各自转为 ImageBind 期望的 (1, 128, 204) mel 谱。
    返回 (actual_clips, 1, 128, 204)，actual_clips 可能小于 clips_per_audio：

      - 短音频（T == clip_samples，已经过 Loop Padding）：
          只取 1 段，避免重复提取相同内容的冗余计算。
      - 长音频（T > clip_samples）：
          均匀取 clips_per_audio 段，起始点分布在整条音频上，
          最后一段若因浮点误差略短则尾部循环补齐。
    """
    T = waveform.shape[1]

    if T <= clip_samples:
        # 短音频经 Loop Padding 后 T == clip_samples，直接取整段，1 次即够
        mel = waveform_to_imagebind_mel(waveform)   # (1, 128, 204)
        return mel.unsqueeze(0)                      # (1, 1, 128, 204)

    clips = []
    for k in range(clips_per_audio):
        start = int(k * (T - clip_samples) / max(clips_per_audio - 1, 1))
        seg   = waveform[:, start: start + clip_samples]
        # 浮点误差兜底：末尾若略短则从头循环补足
        if seg.shape[1] < clip_samples:
            need = clip_samples - seg.shape[1]
            seg  = torch.cat([seg, waveform[:, :need]], dim=1)
        clips.append(waveform_to_imagebind_mel(seg))  # (1, 128, 204)
    return torch.stack(clips, dim=0)                  # (clips_per_audio, 1, 128, 204)


@torch.no_grad()
def extract_audio_features(
    model, dataset: AudioFolderDataset, desc: str,
    clips_per_audio: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    批量提取音频特征，返回 (features: (N,1024), labels: (N,))。

    【Loop Padding + 均匀切段方案】：
      - 短音频（< 2.055s）：循环拼接至恰好 IMAGEBIND_CLIP_SAMPLES=32880 采样点，
        保证频谱图每一帧均有真实声音能量，消除补零特征稀释。
        此时音频等于一个片段长度，clips_per_audio 段退化为 1 段（无冗余计算）。
      - 长音频（>= 2.055s）：保留完整波形，均匀采 clips_per_audio 段取特征均值，
        覆盖音频头、中、尾，增强鲁棒性。
    采样点数由 kaldi fbank 帧数公式精确推导：
      T = (TARGET_LEN-1) × hop_length + frame_length = 203×160+400 = 32880（→ 精确 204 帧）

    clips_per_audio=5：猫音频中存在 Paining/Resting/Mating 等时长 5~17s 的长录音，
    3 段切法在 10s 音频上会留下约 4s 盲区，5 段可将最大盲区压缩到 ~2s，
    更充分覆盖音频内容，对短音频（退化为 1 段）无额外计算开销。
    """
    from imagebind.models.imagebind_model import ModalityType

    TARGET_LEN   = IMAGEBIND_TARGET_LEN    # 204 帧，与预训练位置编码严格对齐
    clip_samples = IMAGEBIND_CLIP_SAMPLES  # 32880 采样点 = (204-1)×160+400，精确 204 帧

    all_feats, all_labels = [], []
    paths  = [dataset.samples[i][0] for i in range(len(dataset))]
    labels = [dataset.samples[i][1] for i in range(len(dataset))]

    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc, ncols=80):
        batch_paths  = [str(p) for p in paths[i: i + BATCH_SIZE]]
        batch_labels = labels[i: i + BATCH_SIZE]

        # mel_clips_list[i]: (actual_clips_i, 1, 128, 204)，短音频为1段，长音频为clips_per_audio段
        mel_clips_list: list[torch.Tensor] = []
        valid_labels = []

        for path, label in zip(batch_paths, batch_labels):
            try:
                waveform = _load_waveform(path, clip_samples)
                clips    = _build_mel_clips(waveform, clip_samples, clips_per_audio)
                mel_clips_list.append(clips)
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"音频读取/处理失败，已动态剔除 {path}: {e}")
        
        if not mel_clips_list:
            continue

        try:
            # 按 clips 数量分组批量推理，保留 GPU 并行，同时兼容 batch 内 clips 数不同
            # 实际上只有两种：短音频→1段，长音频→clips_per_audio段
            from collections import defaultdict
            groups: dict[int, list[int]] = defaultdict(list)
            for idx_in_batch, clips in enumerate(mel_clips_list):
                groups[clips.shape[0]].append(idx_in_batch)

            feat_arr = np.zeros((len(mel_clips_list), 1024), dtype=np.float32)
            for n_clips, idxs in groups.items():
                # (G, n_clips, 1, 128, 204) → (G*n_clips, 1, 128, 204)
                group_tensor = torch.stack([mel_clips_list[j] for j in idxs], dim=0)
                G = group_tensor.shape[0]
                flat = group_tensor.view(G * n_clips, 1, 128, TARGET_LEN).to(DEVICE)
                if DEVICE.type == "cuda":
                    flat = flat.half()
                emb   = model({ModalityType.AUDIO: flat})
                fvecs = emb[ModalityType.AUDIO].float().view(G, n_clips, -1).mean(dim=1)  # (G, 1024)
                # 统一 L2 归一化：与图像特征对齐到同一单位超球面
                fvecs = torch.nn.functional.normalize(fvecs, p=2, dim=-1)
                for out_pos, idx_in_batch in enumerate(idxs):
                    feat_arr[idx_in_batch] = fvecs[out_pos].cpu().numpy()
            feats = feat_arr
        except Exception as e:
            logger.warning(f"批次 {i}~{i+BATCH_SIZE} 模型推理失败，已剔除该批次: {e}")
            continue

        all_feats.append(feats)
        all_labels.extend(valid_labels)

    return np.concatenate(all_feats), np.array(all_labels, dtype=np.int64)


# ============================================================
# 主流程
# ============================================================
def validate_features(feat_dir: Path, prefix: str, n_classes: int):
    """提取完毕后自动校验特征质量：NaN/Inf/零行/范数分布/标签范围。"""
    feat = np.load(feat_dir / f"{prefix}_feat.npy")
    label = np.load(feat_dir / f"{prefix}_label.npy")
    norms = np.linalg.norm(feat, axis=1)

    issues = []
    if np.isnan(feat).any():
        issues.append(f"含 {np.isnan(feat).sum()} 个 NaN 值")
    if np.isinf(feat).any():
        issues.append(f"含 {np.isinf(feat).sum()} 个 Inf 值")
    zero_rows = (norms < 1e-6).sum()
    if zero_rows > 0:
        issues.append(f"含 {zero_rows} 个全零行（模型推理可能失败）")
    if label.min() < 0 or label.max() >= n_classes:
        issues.append(f"标签越界: min={label.min()}, max={label.max()}, 期望 [0, {n_classes})")
    norm_dev = abs(norms.mean() - 1.0)
    if norm_dev > 0.05:
        issues.append(f"L2 范数偏离单位球面: mean={norms.mean():.4f}")

    status = "PASS" if not issues else "FAIL"
    logger.info(f"  校验 [{status}] {prefix}: shape={feat.shape}, "
                f"L2_norm={norms.mean():.4f}+/-{norms.std():.4f}, "
                f"labels={dict(zip(*np.unique(label, return_counts=True)))}")
    for iss in issues:
        logger.warning(f"    ⚠ {iss}")
    return len(issues) == 0


def main():
    import platform

    logger.info("=" * 60)
    logger.info("大创项目 ImageBind 离线特征提取")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"OS: {platform.platform()}")
    logger.info(f"计算设备：{DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"特征输出目录：{FEAT_DIR}")
    logger.info("=" * 60)

    # 加载 ImageBind
    model, _, _ = load_imagebind()

    # Loop Padding + 均匀切段方案：
    # IMAGEBIND_CLIP_SAMPLES = (204-1)×160+400 = 32880，由 kaldi fbank 帧数公式精确推导。
    # 短音频 loop 后取 1 段，长音频均匀取 5 段，猫狗音频统一同一套处理逻辑。
    tasks = [
        # (数据集类, 根目录, 类别列表, 保存前缀, 是否图像)
        (ImageFolderDataset, DOG_IMG_DIR,   DOG_IMG_CLASSES,   "dog_img",   True),
        (ImageFolderDataset, CAT_IMG_DIR,   CAT_IMG_CLASSES,   "cat_img",   True),
        (AudioFolderDataset, DOG_AUDIO_DIR, DOG_AUDIO_CLASSES, "dog_audio", False),
        (AudioFolderDataset, CAT_AUDIO_DIR, CAT_AUDIO_CLASSES, "cat_audio", False),
    ]

    for DatasetClass, root, classes, prefix, is_image in tasks:
        logger.info(f"\n{'─'*40}")
        logger.info(f"开始处理：{prefix}  ({root.name})")
        t0 = time.time()

        dataset = DatasetClass(root, classes)
        if len(dataset) == 0:
            logger.error(f"  数据为空，跳过！请检查路径：{root}")
            continue

        desc = f"[{prefix}]"
        if is_image:
            feats, labels = extract_image_features(model, dataset, desc)
        else:
            feats, labels = extract_audio_features(model, dataset, desc)

        feat_path  = FEAT_DIR / f"{prefix}_feat.npy"
        label_path = FEAT_DIR / f"{prefix}_label.npy"
        np.save(feat_path,  feats)
        np.save(label_path, labels)

        elapsed = time.time() - t0
        logger.info(f"  已保存：{feat_path.name}  shape={feats.shape}")
        logger.info(f"  已保存：{label_path.name} shape={labels.shape}")
        logger.info(f"  耗时：{elapsed:.1f} 秒")

    # ── 特征质量校验 ──────────────────────────────────────────
    logger.info("\n" + "─" * 40)
    logger.info("特征质量自动校验：")
    all_pass = True
    for _, root, classes, prefix, _ in tasks:
        if not (FEAT_DIR / f"{prefix}_feat.npy").exists():
            continue
        ok = validate_features(FEAT_DIR, prefix, len(classes))
        all_pass = all_pass and ok

    logger.info("\n" + "=" * 60)
    if all_pass:
        logger.info("全部特征提取完毕，质量校验全部通过！")
    else:
        logger.warning("特征提取完毕，但部分校验未通过，请检查上方警告信息。")
    logger.info(f"日志已保存：{log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
