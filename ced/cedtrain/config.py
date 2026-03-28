# -*- coding: utf-8 -*-
"""
集中配置：超参数、路径、标签映射。

所有可调参数在此统一管理，其余模块仅 import 本文件，
确保实验可复现、参数变更一处修改全局生效。
"""

from pathlib import Path
from dataclasses import dataclass, field

# ============================================================
# 路径配置
# ============================================================
ROOT = Path(__file__).resolve().parent.parent

CED_MODEL_DIR = ROOT / "pretrained_modelscope" / "ced-mini"
DOG_AUDIO_DIR = ROOT / "data" / "Pet dog sound event"
CAT_AUDIO_DIR = ROOT / "data" / "CatSound_DataSet_V2" / "NAYA_DATA_AUG1X"

MODEL_DIR = ROOT / "moxing"
FIG_DIR   = ROOT / "figure"
TXT_DIR   = ROOT / "txt"

# ============================================================
# 标签体系
# ============================================================
DOG_AUDIO_CLASSES = ["barking", "growling", "howling", "whining"]
CAT_AUDIO_CLASSES = [
    "Angry", "Defence", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning",
]

TASK_META = {
    "dog_audio": {
        "root": DOG_AUDIO_DIR,
        "classes": DOG_AUDIO_CLASSES,
        "num_classes": len(DOG_AUDIO_CLASSES),
    },
    "cat_audio": {
        "root": CAT_AUDIO_DIR,
        "classes": CAT_AUDIO_CLASSES,
        "num_classes": len(CAT_AUDIO_CLASSES),
    },
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


# ============================================================
# 训练超参数
# ============================================================
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 60
    patience: int = 10

    # 学习率
    backbone_lr: float = 2e-5          # 顶层 block 骨干 LR
    backbone_lr_decay: float = 0.5     # 每往下一层 block LR × 此系数（层级衰减）
    head_lr: float = 5e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01

    # 渐进式解冻调度：{起始 epoch: 解冻策略名称}
    # "head_only"    → 骨干全冻，仅训练投影/ResBlock/分类头
    # "top2"         → 解冻顶层 2 个 block (block 10-11) + LN
    # "top4"         → 解冻顶层 4 个 block (block 8-11) + LN（训练后期充分微调）
    unfreeze_schedule: dict = field(default_factory=lambda: {
        1:  "head_only",   # Epoch 1~warmup_epochs: 头部收敛，保护预训练特征
        6:  "top2",        # warmup 后先解冻顶 2 层（与 warmup_epochs=5 对齐）
        16: "top4",        # 中后期全面解冻顶 4 层
    })

    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # Focal Loss
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # Mixup（特征空间）
    mixup_alpha: float = 0.3

    # 物种（狗/猫）二分类损失权重；与 train/3 中 species 项一致，与情绪任务加权和
    species_loss_weight: float = 0.3

    # 分类头 Dropout（狗音频样本常少于猫 → 狗分支略强正则，与 train/3 少样本侧思路一致）
    task_dropouts: dict = field(default_factory=lambda: {
        "dog_audio": 0.32,
        "cat_audio": 0.26,
    })

    # 训练期仅对 dog_audio 行施加的 Mel 高斯噪声（log-mel 域）；0 关闭。对标「少数侧更强增强」
    dog_audio_mel_noise_std: float = 0.02

    # SpecAugment（log-mel 域，训练期）；与 AST/BEATs 等一致，提升小数据泛化
    spec_augment_enabled: bool = True
    spec_aug_freq_param: int = 13   # 单条频率掩码最大宽度（mel 维）
    spec_aug_time_param: int = 30   # 单条时间掩码最大宽度（帧）
    spec_aug_num_freq: int = 2
    spec_aug_num_time: int = 2

    # EMA：对参数指数滑动平均，验证/早停/保存最优权重默认用 EMA 权重（关闭则与旧行为一致）
    use_ema: bool = True
    ema_decay: float = 0.999

    # CED-Mini 音频参数（与 preprocessor_config.json 严格对齐）
    sampling_rate: int = 16000
    n_fft: int = 512
    win_size: int = 512
    hop_size: int = 160
    n_mels: int = 64
    f_min: int = 0
    f_max: int = 8000
    # Mel 动态范围上界，与 AmplitudeToDB(top_db=mel_top_db) 一致；padding 用 -mel_top_db
    mel_top_db: int = 120

    # 音频预处理
    max_audio_sec: float = 10.0
    clip_sec: float = 10.0

    # TTA：多视角（噪声 / 时间平移 / 轻量频域掩码 / 轻量时域掩码）轮换，步数仍为 tta_steps
    tta_steps: int = 10
    tta_noise_std: float = 0.02
    tta_time_shift_max: int = 8       # 沿时间维平移最大帧数（torch.roll）
    tta_freq_mask_max: int = 8        # 单步 TTA 频率掩码最大宽度
    tta_time_mask_max: int = 12       # 单步 TTA 时间掩码最大宽度

    # 混合精度
    use_amp: bool = True

    # 梯度裁剪
    max_grad_norm: float = 5.0

    @property
    def clip_samples(self) -> int:
        return int(self.clip_sec * self.sampling_rate)

    @property
    def max_audio_samples(self) -> int:
        return int(self.max_audio_sec * self.sampling_rate)

    @property
    def mel_pad_db(self) -> float:
        """与 AmplitudeToDB 后静音帧对应的 dB 值，用于 batch 右侧 padding。"""
        return -float(self.mel_top_db)
