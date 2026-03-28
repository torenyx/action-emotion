# -*- coding: utf-8 -*-
"""
工具函数：种子固定、日志初始化、环境信息记录。
"""

import os
import sys
import logging
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.font_manager as fm

from .config import TXT_DIR


def get_zh_font() -> fm.FontProperties:
    """matplotlib 中文字体（与 visualization 模块一致）。"""
    zh_fonts = [
        f.fname for f in fm.fontManager.ttflist
        if any(
            kw in f.name
            for kw in ("SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong")
        )
    ]
    if zh_fonts:
        return fm.FontProperties(fname=zh_fonts[0])
    return fm.FontProperties()


def seed_everything(seed: int) -> None:
    """全局可复现性：固定所有随机源。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str, prefix: str = "ced_train") -> tuple[logging.Logger, Path]:
    """
    创建双输出（终端 + 文件）的 logger。
    返回 (logger, log_path)。
    """
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = TXT_DIR / f"{prefix}_log_{ts}.txt"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger, log_path


def log_environment(logger: logging.Logger) -> None:
    """记录运行环境关键信息，便于论文复现与 debug。"""
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"OS: {platform.platform()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"GPU 显存: {mem_gb:.1f} GB")
    else:
        logger.info("未检测到 CUDA，将使用 CPU 训练")


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")
