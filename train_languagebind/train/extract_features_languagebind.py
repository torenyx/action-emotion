# -*- coding: utf-8 -*-
"""
大创项目：LanguageBind 离线特征提取脚本（工业级）

功能：
  - 加载 LanguageBind Image / Audio-FT 预训练权重（从 HuggingFace）
  - 遍历四类数据（猫图/狗图/猫音频/狗音频），批量提取 768 维特征
  - 将特征与标签保存为 .npy 文件，供后续 MLP 训练极速复用
  - 支持 fp16 半精度推理，RTX 4060 (8GB) 可完整运行

与 ImageBind 方案的主要差异：
  - 特征维度：768（LanguageBind projection_dim）vs 1024（ImageBind）
  - 图像编码器：OpenCLIP ViT-L/14 初始化（与 ImageBind 相同骨干但不同训练策略）
  - 音频编码器：独立 ViT + VIDAL-10M 全量微调，mel 参数不同
    num_mel_bins=112, target_length=1036（vs ImageBind 的 128×204）
  - 音频预处理：LanguageBind 自带 AudioProcessor，无需手动构建 kaldi fbank

运行方式：
  conda activate d2l
  python train/extract_features_languagebind.py

输出：
  data/features_languagebind_npy/
    dog_img_feat.npy   (N, 768)
    dog_img_label.npy  (N,)
    cat_img_feat.npy   (M, 768)
    cat_img_label.npy  (M,)
    dog_audio_feat.npy (P, 768)
    dog_audio_label.npy(P,)
    cat_audio_feat.npy (Q, 768)
    cat_audio_label.npy(Q,)
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 将本地克隆的 LanguageBind 仓库加入路径（无需 pip install）
_LB_REPO = Path(__file__).resolve().parent.parent / "LanguageBind"
if str(_LB_REPO) not in sys.path:
    sys.path.insert(0, str(_LB_REPO))

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
FEAT_DIR = ROOT / "data" / "features_languagebind_npy"
TXT_DIR = ROOT / "txt"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d%H%M%S")
log_path = TXT_DIR / f"extract_features_lb_{TS}.txt"

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

for handler in logging.root.handlers:
    handler.addFilter(WarningFilter())

# ============================================================
# 数据路径与标签定义（与 ImageBind 版完全一致）
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

# LanguageBind 音频参数（来自 config.json）
LB_AUDIO_SR          = 16000
LB_AUDIO_NUM_MELS    = 112
LB_AUDIO_TARGET_LEN  = 1036
LB_AUDIO_MEAN        = -4.2677393
LB_AUDIO_STD         = 4.5689974

# 超长音频截断保护（120 秒）
MAX_AUDIO_SEC     = 120
MAX_AUDIO_SAMPLES = MAX_AUDIO_SEC * LB_AUDIO_SR

# LanguageBind Audio 的 mel 谱对应的原始采样点数
# ViT patch_size=14, image_size=224 → 224/14=16 个 patch（每行/列）
# target_length=1036 帧，对应约 10 秒音频（10s × 16000Hz = 160000 采样点）
# hop_length = 160（10ms），frame_length = 400（25ms）
LB_AUDIO_HOP_LENGTH    = 160
LB_AUDIO_FRAME_LENGTH  = 400
LB_AUDIO_CLIP_SAMPLES  = (LB_AUDIO_TARGET_LEN - 1) * LB_AUDIO_HOP_LENGTH + LB_AUDIO_FRAME_LENGTH

BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM  = 768

# ============================================================
# LanguageBind 加载工具
# ============================================================
def load_languagebind_image():
    """加载 LanguageBind Image 编码器（基于 OpenCLIP ViT-L/14）。"""
    try:
        from languagebind import (LanguageBindImage, LanguageBindImageTokenizer,
                                  LanguageBindImageProcessor)
        logger.info("检测到本地 languagebind 包，正在加载 Image 模型...")
    except ImportError:
        logger.error(
            "未安装 languagebind！请执行：\n"
            "  pip install languagebind\n"
            "或从源码安装：\n"
            "  git clone https://github.com/PKU-YuanGroup/LanguageBind\n"
            "  cd LanguageBind && pip install -e ."
        )
        sys.exit(1)

    pretrained_ckpt = "LanguageBind/LanguageBind_Image"
    cache_dir = str(ROOT / "cache_dir")

    logger.info(f"正在从 HuggingFace 加载/下载模型权重：{pretrained_ckpt}")
    logger.info(f"缓存目录：{cache_dir}（若首次运行需下载约 1-2GB，请耐心等待）")
    logger.info("下载进度条将显示在终端，若无进度条说明已命中缓存正在加载权重...")
    model = LanguageBindImage.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
    logger.info("Image 模型权重加载完成，正在加载 tokenizer...")
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
    logger.info("Image tokenizer 加载完成，正在构建 processor...")
    processor = LanguageBindImageProcessor(model.config, tokenizer)

    model.eval()
    if DEVICE.type == "cuda":
        model = model.half().to(DEVICE)
    else:
        model = model.to(DEVICE)

    logger.info(f"LanguageBind Image 加载完成，运行在 {DEVICE}"
                f"（{'fp16' if DEVICE.type == 'cuda' else 'fp32'}）")
    return model, processor


def load_languagebind_audio():
    """加载 LanguageBind Audio-FT 编码器（全量微调版，性能更强）。"""
    try:
        from languagebind import (LanguageBindAudio, LanguageBindAudioTokenizer,
                                  LanguageBindAudioProcessor)
        logger.info("正在加载 LanguageBind Audio-FT 模型...")
    except ImportError:
        logger.error("未安装 languagebind！请参考上方提示安装。")
        sys.exit(1)

    pretrained_ckpt = "LanguageBind/LanguageBind_Audio_FT"
    cache_dir = str(ROOT / "cache_dir")

    logger.info(f"正在从 HuggingFace 加载/下载模型权重：{pretrained_ckpt}")
    logger.info(f"缓存目录：{cache_dir}（若首次运行需下载约 1-2GB，请耐心等待）")
    logger.info("下载进度条将显示在终端，若无进度条说明已命中缓存正在加载权重...")
    model = LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
    logger.info("Audio 模型权重加载完成，正在加载 tokenizer...")
    tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
    logger.info("Audio tokenizer 加载完成，正在构建 processor...")
    processor = LanguageBindAudioProcessor(model.config, tokenizer)

    model.eval()
    if DEVICE.type == "cuda":
        model = model.half().to(DEVICE)
    else:
        model = model.to(DEVICE)

    logger.info(f"LanguageBind Audio-FT 加载完成，运行在 {DEVICE}"
                f"（{'fp16' if DEVICE.type == 'cuda' else 'fp32'}）")
    return model, processor


# ============================================================
# 图像预处理（与 ImageBind 版对齐：PadToSquare 保留完整面部特征）
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
        padding = (0, pad1, 0, pad2) if w > h else (pad1, 0, pad2, 0)
        return transforms.functional.pad(img, padding, fill=self.fill, padding_mode='constant')


# LanguageBind Image 与 ImageBind 共享同一套 OpenAI CLIP 归一化常量
_LB_IMG_TRANSFORM = transforms.Compose([
    PadToSquare(fill=128),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


# ============================================================
# 图像 Dataset（手动预处理，绕过 Processor 的 CenterCrop）
# ============================================================
class ImageFolderDataset(Dataset):
    """从 cls_dir/class_name/*.jpg 结构扫描图像并完成预处理。"""

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
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (128, 128, 128))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")
                img = _LB_IMG_TRANSFORM(img)
            return img, label
        except Exception as e:
            logger.warning(f"读取图像数据失败已动态剔除 {path}: {e}")
            return None


# ============================================================
# 音频 Dataset
# ============================================================
class AudioFolderDataset(Dataset):
    """从 cls_dir/class_name/*.wav 结构扫描音频，仅收集路径与标签。"""

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
# 图像 collate（过滤损坏样本）
# ============================================================
def collate_drop_none(batch):
    """过滤掉 __getitem__ 返回 None 的损坏样本"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    return imgs, labels


# ============================================================
# 图像特征提取（绕过 Processor，使用 PadToSquare 预处理）
# ============================================================
@torch.no_grad()
def extract_image_features(
    model, processor, dataset: ImageFolderDataset, desc: str,
    multi_view: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    使用 LanguageBind Image 批量提取图像特征，返回 (features: (N,768), labels: (N,))。

    【关键修复】绕过 LanguageBindImageProcessor 内置的 CenterCrop 预处理，
    改用 PadToSquare → Resize(224) 流程，与 ImageBind 版对齐。
    CenterCrop 会裁掉非正方形图片的边缘（猫的耳朵、下巴等承载情绪信息的关键区域），
    导致 cat_img（667张/7类）的细粒度情绪特征严重退化。

    multi_view=True 时，对每张图提取原图 + 水平翻转两个视角的特征并取均值，
    产出更鲁棒的、对左右方向不变的表征。
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
                        collate_fn=collate_drop_none)
    all_feats, all_labels = [], []

    for batch in tqdm(loader, desc=desc, ncols=80):
        if batch[0] is None:
            continue
        imgs, labels = batch

        if DEVICE.type == "cuda":
            pixel_values = imgs.half().to(DEVICE)
        else:
            pixel_values = imgs.to(DEVICE)

        emb_orig = model.get_image_features(pixel_values=pixel_values)

        if multi_view:
            pixel_flip = torch.flip(pixel_values, dims=[-1])
            emb_flip = model.get_image_features(pixel_values=pixel_flip)
            feats = ((emb_orig + emb_flip) / 2.0).cpu().float()
        else:
            feats = emb_orig.cpu().float()

        feats = torch.nn.functional.normalize(feats, p=2, dim=-1).numpy()
        all_feats.append(feats)
        all_labels.append(labels.numpy())

    return np.concatenate(all_feats), np.concatenate(all_labels)


# ============================================================
# 音频特征提取
# ============================================================
def _build_audio_transform(audio_model):
    """
    从模型 config 构建 LanguageBind AudioTransform，完全绕过 Processor 的
    size 校验逻辑（该逻辑错误地将矩形 mel 谱当正方形图像校验，导致全量失败）。
    参考 ImageBind 的处理方式：手动预处理后直接传 pixel_values 给模型。
    """
    from languagebind.audio.processing_audio import AudioTransform
    return AudioTransform(audio_model.config.vision_config)


def _wav_to_mel_fusion(path: str, transform) -> torch.Tensor:
    """
    读取单条音频，经 AudioTransform 转为 mel fusion 张量。
    输出形状：(3, num_mel_bins, target_length) = (3, 112, 1036)

    对超长音频（>MAX_AUDIO_SEC 秒）先截断，防止内存溢出；
    对超响音频做峰值归一化，消除响度偏差（同 ImageBind 处理）。
    """
    waveform, sr = torchaudio.load(path)

    # 超长截断
    max_samples = MAX_AUDIO_SAMPLES
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # 峰值归一化：消除部分 MP3/WAV 过载编码导致振幅超 1.0
    peak = waveform.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        waveform = waveform / peak

    # AudioTransform 内部会自动重采样到 16kHz，并返回 (3, num_mel_bins, target_length)
    mel_fusion = transform((waveform, sr))
    return mel_fusion  # (3, 112, 1036)


@torch.no_grad()
def extract_audio_features(
    model, processor, dataset: AudioFolderDataset, desc: str,
    clips_per_audio: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    使用 LanguageBind Audio-FT 批量提取音频特征，返回 (features: (N,768), labels: (N,))。

    【关键修复】绕过 LanguageBindAudioProcessor 传路径的方式——该方式内部调用了
    transformers CLIPVisionEmbeddings 的正方形 image_size 校验逻辑，
    对 (112, 1036) 矩形 mel 谱全量报错 "Input image size doesn't match model"。

    改为参考 ImageBind 的做法：直接调用底层 AudioTransform 手动生成
    (3, 112, 1036) mel fusion 张量，再以 pixel_values 传入模型，
    完全绕开有 bug 的 size 校验路径。
    """
    # 构建底层 AudioTransform（绕过 Processor）
    audio_transform = _build_audio_transform(model)

    all_feats, all_labels = [], []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=desc, ncols=80):
        batch_items = []
        for j in range(i, min(i + BATCH_SIZE, len(dataset))):
            path, label = dataset[j]
            batch_items.append((path, label))

        valid_mels = []
        valid_labels = []

        for path, label in batch_items:
            try:
                mel = _wav_to_mel_fusion(path, audio_transform)  # (3, 112, 1036)
                valid_mels.append(mel)
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"音频处理失败，已跳过 {path}: {e}")
                continue

        if not valid_mels:
            continue

        try:
            # 拼成批次：(B, 3, 112, 1036)
            pixel_values = torch.stack(valid_mels, dim=0)
            if DEVICE.type == "cuda":
                pixel_values = pixel_values.half().to(DEVICE)
            else:
                pixel_values = pixel_values.to(DEVICE)

            emb = model.get_image_features(pixel_values=pixel_values)
            feats = emb.cpu().float()
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1).numpy()

            all_feats.append(feats)
            all_labels.extend(valid_labels)

        except Exception as e:
            logger.warning(f"批次 {i}~{i+BATCH_SIZE} 模型推理失败，已跳过: {e}")
            continue

    if not all_feats:
        raise RuntimeError(
            "音频特征提取失败：所有样本均处理失败，请检查音频文件格式与模型配置。"
        )

    return np.concatenate(all_feats), np.array(all_labels, dtype=np.int64)


# ============================================================
# 特征质量校验
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

    status = "通过" if not issues else "失败"
    logger.info(f"  校验 [{status}] {prefix}: shape={feat.shape}, "
                f"L2_norm={norms.mean():.4f}+/-{norms.std():.4f}, "
                f"labels={dict(zip(*np.unique(label, return_counts=True)))}")
    for iss in issues:
        logger.warning(f"    ⚠ {iss}")
    return len(issues) == 0


# ============================================================
# 主流程
# ============================================================
def main():
    import platform

    logger.info("=" * 60)
    logger.info("大创项目 LanguageBind 离线特征提取")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"OS: {platform.platform()}")
    logger.info(f"计算设备：{DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"特征维度：{EMBED_DIM}（LanguageBind projection_dim）")
    logger.info(f"特征输出目录：{FEAT_DIR}")
    logger.info("=" * 60)

    # ── 加载 LanguageBind Image 和 Audio ─────────────────────
    img_model, img_processor = load_languagebind_image()
    audio_model, audio_processor = load_languagebind_audio()

    tasks = [
        # (数据集类, 根目录, 类别列表, 保存前缀, 是否图像, 模型, 处理器)
        (ImageFolderDataset, DOG_IMG_DIR,   DOG_IMG_CLASSES,   "dog_img",   True,  img_model, img_processor),
        (ImageFolderDataset, CAT_IMG_DIR,   CAT_IMG_CLASSES,   "cat_img",   True,  img_model, img_processor),
        (AudioFolderDataset, DOG_AUDIO_DIR, DOG_AUDIO_CLASSES, "dog_audio", False, audio_model, audio_processor),
        (AudioFolderDataset, CAT_AUDIO_DIR, CAT_AUDIO_CLASSES, "cat_audio", False, audio_model, audio_processor),
    ]

    for DatasetClass, root, classes, prefix, is_image, model, processor in tasks:
        logger.info(f"\n{'─'*40}")
        logger.info(f"开始处理：{prefix}  ({root.name})")
        t0 = time.time()

        dataset = DatasetClass(root, classes)
        if len(dataset) == 0:
            logger.error(f"  数据为空，跳过！请检查路径：{root}")
            continue

        desc = f"[{prefix}]"
        if is_image:
            feats, labels = extract_image_features(model, processor, dataset, desc)
        else:
            feats, labels = extract_audio_features(model, processor, dataset, desc)

        feat_path  = FEAT_DIR / f"{prefix}_feat.npy"
        label_path = FEAT_DIR / f"{prefix}_label.npy"
        np.save(feat_path,  feats)
        np.save(label_path, labels)

        elapsed = time.time() - t0
        logger.info(f"  已保存：{feat_path.name}  shape={feats.shape}")
        logger.info(f"  已保存：{label_path.name} shape={labels.shape}")
        logger.info(f"  耗时：{elapsed:.1f} 秒")

    # ── 释放 GPU 显存 ──────────────────────────────────────
    del img_model, audio_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── 特征质量校验 ──────────────────────────────────────
    logger.info("\n" + "─" * 40)
    logger.info("特征质量自动校验：")
    all_pass = True
    for _, root, classes, prefix, _, _, _ in tasks:
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
