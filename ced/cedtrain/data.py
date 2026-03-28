# -*- coding: utf-8 -*-
"""
数据管线：音频扫描、波形预处理、CED-Mini Mel 特征提取、Dataset / DataLoader 构建。

设计原则：
  - 音频预处理与 CED-Mini 官方 Feature Extractor 完全对齐
    (16kHz, 64-mel, n_fft=512, hop=160, AmplitudeToDB top_db=120)
  - 短音频 Loop Padding，长音频截断至 max_audio_sec
  - 峰值归一化至 [-1,1] 量级，消除响度与编码差异
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

from .config import TrainConfig, TASK_META, AUDIO_EXTENSIONS

logger = logging.getLogger("cedtrain")

# 重采样器缓存，避免每条样本重复构造 Resample
_RESAMPLE_CACHE: dict[tuple[int, int], T.Resample] = {}


def _get_resample(orig_sr: int, target_sr: int) -> T.Resample:
    key = (orig_sr, target_sr)
    if key not in _RESAMPLE_CACHE:
        _RESAMPLE_CACHE[key] = T.Resample(orig_sr, target_sr)
    return _RESAMPLE_CACHE[key]


# ============================================================
# 波形 I/O
# ============================================================

def load_waveform(path: str, cfg: TrainConfig) -> torch.Tensor:
    """
    读取音频 → 重采样到 16kHz → 单声道 → 峰值归一化 → Loop Padding/截断。
    返回 (1, T') 张量，T' >= clip_samples。
    """
    waveform, sr = torchaudio.load(path)
    if sr != cfg.sampling_rate:
        waveform = _get_resample(sr, cfg.sampling_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    t_len = waveform.shape[1]
    if t_len == 0:
        raise ValueError(f"空音频: {path}")

    if t_len > cfg.max_audio_samples:
        waveform = waveform[:, : cfg.max_audio_samples]
        t_len = cfg.max_audio_samples

    peak = waveform.abs().max().clamp(min=1e-8)
    waveform = waveform / peak

    if t_len < cfg.clip_samples:
        repeats = (cfg.clip_samples + t_len - 1) // t_len
        waveform = torch.tile(waveform, (1, repeats))
        waveform = waveform[:, : cfg.clip_samples]

    return waveform


# ============================================================
# Mel 频谱构建（与 CED 官方 FeatureExtractor 严格一致）
# ============================================================

def waveform_to_mel(
    waveform: torch.Tensor,
    mel_spectrogram: T.MelSpectrogram,
    amp_to_db: T.AmplitudeToDB,
) -> torch.Tensor:
    """
    将 (1, T) 波形转为 CED-Mini 期望的 (n_mels, n_frames) log-mel 频谱图。
    与 feature_extraction_ced.py 中 CedFeatureExtractor 完全一致：
      MelSpectrogram → AmplitudeToDB
    """
    mel = mel_spectrogram(waveform)   # (1, n_mels, n_frames)
    mel = amp_to_db(mel)              # (1, n_mels, n_frames)
    return mel.squeeze(0)           # (n_mels, n_frames)


# ============================================================
# 音频文件夹 Dataset
# ============================================================

class AudioEmotionDataset(Dataset):
    """
    联合猫狗音频数据集。

    每条样本返回 dict:
      - mel:      (n_mels, n_frames) log-mel 频谱
      - label:    类别索引 (在当前 task 内)
      - task_id:  0=dog_audio, 1=cat_audio
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.samples: list[tuple[str, int, int]] = []   # (path, label, task_id)
        self.task_class_names: dict[str, list[str]] = {}

        for task_id, (task_name, meta) in enumerate(TASK_META.items()):
            root = meta["root"]
            classes = meta["classes"]
            self.task_class_names[task_name] = classes
            skipped = 0

            for label_idx, cls_name in enumerate(classes):
                cls_dir = Path(root) / cls_name
                if not cls_dir.exists():
                    logger.warning(f"类别目录不存在: {cls_dir}")
                    continue
                for p in cls_dir.iterdir():
                    if p.suffix.lower() not in AUDIO_EXTENSIONS:
                        continue
                    try:
                        torchaudio.info(str(p))
                        self.samples.append((str(p), label_idx, task_id))
                    except Exception:
                        skipped += 1

            n_task = sum(1 for _, _, tid in self.samples if tid == task_id)
            logger.info(
                f"  加载 {task_name}: {n_task} 条音频, {len(classes)} 类"
                + (f" (剔除损坏 {skipped} 条)" if skipped else "")
            )

        self.labels = np.array([s[1] for s in self.samples], dtype=np.int64)
        self.task_ids = np.array([s[2] for s in self.samples], dtype=np.int64)
        self.stratify_labels = self.task_ids * 100 + self.labels

        logger.info(f"  数据集合并完毕: 共 {len(self.samples)} 条样本")

        self._mel_spectrogram = T.MelSpectrogram(
            sample_rate=cfg.sampling_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_size,
            hop_length=cfg.hop_size,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            center=True,
        )
        self._amp_to_db = T.AmplitudeToDB(top_db=cfg.mel_top_db)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """读取失败时切换到下一个索引重试，避免「静音频谱 + 真实标签」污染梯度。"""
        n = len(self.samples)
        start_idx = idx
        last_err: Exception | None = None
        max_retries = min(32, n)
        for attempt in range(max_retries):
            path, label, task_id = self.samples[idx]
            try:
                waveform = load_waveform(path, self.cfg)
                mel = waveform_to_mel(waveform, self._mel_spectrogram, self._amp_to_db)
                return {
                    "mel": mel,
                    "label": torch.tensor(label, dtype=torch.long),
                    "task_id": torch.tensor(task_id, dtype=torch.long),
                }
            except Exception as e:
                last_err = e
                logger.warning(
                    f"音频处理失败 {path}: {e}，切换索引重试 ({attempt + 1}/{max_retries})",
                )
                idx = (idx + 1) % n
        raise RuntimeError(
            f"无法从数据集读取有效 mel（起始索引={start_idx}，"
            f"尝试 {max_retries} 个索引均失败），最后错误: {last_err}",
        ) from last_err


# ============================================================
# Collate：pad mel 到 batch 内最大长度
# ============================================================

class MelCollateFn:
    """
    动态 padding: 将 batch 内所有 mel 频谱 pad 到相同帧数。
    padding 值与 AmplitudeToDB(mel_top_db) 下静音一致（见 cfg.mel_pad_db）。

    使用类而非嵌套函数，以便在 Windows（spawn）下 DataLoader(num_workers>0) 能 pickle collate。
    """

    __slots__ = ("pad_val",)

    def __init__(self, pad_val: float):
        self.pad_val = pad_val

    def __call__(self, batch: list[dict]) -> dict:
        max_frames = max(item["mel"].shape[1] for item in batch)

        padded_mels = []
        for item in batch:
            mel = item["mel"]
            pad_len = max_frames - mel.shape[1]
            if pad_len > 0:
                mel = torch.nn.functional.pad(mel, (0, pad_len), value=self.pad_val)
            padded_mels.append(mel)

        return {
            "mel": torch.stack(padded_mels),
            "label": torch.stack([item["label"] for item in batch]),
            "task_id": torch.stack([item["task_id"] for item in batch]),
        }


def make_collate_fn(cfg: TrainConfig) -> MelCollateFn:
    return MelCollateFn(cfg.mel_pad_db)


# ============================================================
# 数据划分 & 采样器
# ============================================================

def stratified_split(
    dataset: AudioEmotionDataset,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于 (task_id, label) 联合标签做分层采样。"""
    labels = dataset.stratify_labels
    n = len(labels)
    test_ratio = 1.0 - train_ratio - val_ratio

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(n), labels))

    relative_val = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=seed,
    )
    train_sub, val_sub = next(
        sss2.split(np.zeros(len(trainval_idx)), labels[trainval_idx]),
    )

    return trainval_idx[train_sub], trainval_idx[val_sub], test_idx


def make_weighted_sampler(
    dataset: AudioEmotionDataset, indices: np.ndarray,
) -> WeightedRandomSampler:
    """逆频率过采样，使各类被采到的期望次数近似相等。"""
    strat = dataset.stratify_labels[indices]
    counter = Counter(strat)
    weights = np.array([1.0 / counter[s] for s in strat], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)


def build_dataloaders(
    dataset: AudioEmotionDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: TrainConfig,
    device_type: str = "cuda",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train/val/test 三个 DataLoader。"""
    pin = device_type == "cuda"
    collate_fn = make_collate_fn(cfg)
    loader_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=pin,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    train_sampler = make_weighted_sampler(dataset, train_idx)
    train_loader = DataLoader(
        Subset(dataset, train_idx), sampler=train_sampler, **loader_kw,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kw)
    test_loader = DataLoader(Subset(dataset, test_idx), shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader
