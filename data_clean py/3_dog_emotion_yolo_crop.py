# -*- coding: utf-8 -*-
"""
对 data_raw/dog/Dog Emotion 下四个情绪子文件夹（angry / happy / relaxed / sad）
逐张做狗目标检测并扩边裁剪。

流程与 2_yolo_crop 一致：首模推理，首模未检出狗框时再跑 YOLOv11 补检。
本数据集仅采纳 COCO 狗类 id=16，不采纳猫框。

输出：
  - data/dog_emotion_cropped：检出成功后的裁剪或整图（四个子文件夹）
  - data/dog_emotion_cropped_no_detection：仅「双模均未检出」的整图（不写入主目录）
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────────
# 配置项
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC_BASE = ROOT / "data_raw" / "dog" / "Dog Emotion"
OUT_DIR = ROOT / "data" / "dog_emotion_cropped"
NO_DET_DIR = ROOT / "data" / "dog_emotion_cropped_no_detection"

# 仅狗（狗情绪数据，不采纳猫框）
DOG_CLASS_IDS: set[int] = {16}

YOLO_MODEL_PRIMARY = "yolo26x.pt"
YOLO_MODEL_V11 = "yolo11x.pt"
CONF_THRESHOLD = 0.10
PADDING_RATIO = 0.03
FULLIMG_RATIO = 0.98
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# 只处理这四个文件夹，顺序固定便于对照
EMOTION_SUBDIRS: tuple[str, ...] = ("angry", "happy", "relaxed", "sad")

LOG_DIR = ROOT / "txt"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now().strftime("%Y%m%d%H%M%S")
log_path = LOG_DIR / f"dog_emotion_yolo_crop_{TS}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def expand_box(x1: int, y1: int, x2: int, y2: int,
               img_w: int, img_h: int, ratio: float
               ) -> tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * ratio)
    pad_y = int(bh * ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_w, x2 + pad_x)
    ny2 = min(img_h, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def pick_best_dog_box(results, dog_ids: set[int], conf_thresh: float
                      ) -> tuple[int, int, int, int] | None:
    """取置信度最高的狗框；无则 None。"""
    best_conf = conf_thresh
    best_box = None
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id in dog_ids and conf >= best_conf:
            best_conf = conf
            xyxy = box.xyxy[0].tolist()
            best_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
    return best_box


def infer_dog_box(model: YOLO, img, dog_ids: set[int], conf_thresh: float
                  ) -> tuple[int, int, int, int] | None:
    results = model(img, verbose=False)
    return pick_best_dog_box(results, dog_ids, conf_thresh)


def main() -> None:
    if not SRC_BASE.is_dir():
        raise FileNotFoundError(f"源目录不存在：{SRC_BASE}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if NO_DET_DIR.exists():
        shutil.rmtree(NO_DET_DIR)
    OUT_DIR.mkdir(parents=True)
    NO_DET_DIR.mkdir(parents=True)

    logger.info("加载首模：%s；补检：%s", YOLO_MODEL_PRIMARY, YOLO_MODEL_V11)
    model_primary = YOLO(YOLO_MODEL_PRIMARY)
    model_v11 = YOLO(YOLO_MODEL_V11)

    for name in EMOTION_SUBDIRS:
        p = SRC_BASE / name
        if not p.is_dir():
            raise FileNotFoundError(f"缺少子文件夹：{p}")

    total = 0
    cropped = 0
    fullimg = 0                 # 主目录整图（大头照；双模未检出不计入）
    no_detection_final = 0
    v11_recovered = 0
    skipped = 0
    per_class_stats: dict[str, dict[str, int]] = {}

    for cls_name in EMOTION_SUBDIRS:
        cls_dir = SRC_BASE / cls_name
        out_cls = OUT_DIR / cls_name
        no_det_cls = NO_DET_DIR / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)
        no_det_cls.mkdir(parents=True, exist_ok=True)

        img_paths = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]

        c_crop = c_fullimg = c_skip = 0
        c_no_det_final = 0
        c_v11_ok = 0

        for img_path in img_paths:
            total += 1
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("无法读取图片，跳过：%s/%s", cls_name, img_path.name)
                c_skip += 1
                skipped += 1
                continue

            img_h, img_w = img.shape[:2]
            img_area = img_h * img_w

            box = infer_dog_box(model_primary, img, DOG_CLASS_IDS, CONF_THRESHOLD)
            if box is None:
                box = infer_dog_box(model_v11, img, DOG_CLASS_IDS, CONF_THRESHOLD)
                if box is not None:
                    c_v11_ok += 1
                    v11_recovered += 1

            out_path = out_cls / img_path.name

            if box is not None:
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / img_area >= FULLIMG_RATIO:
                    cv2.imwrite(str(out_path), img)
                    c_fullimg += 1
                    fullimg += 1
                else:
                    ex1, ey1, ex2, ey2 = expand_box(
                        x1, y1, x2, y2, img_w, img_h, PADDING_RATIO,
                    )
                    crop = img[ey1:ey2, ex1:ex2]
                    cv2.imwrite(str(out_path), crop)
                    c_crop += 1
                    cropped += 1
            else:
                no_detection_final += 1
                c_no_det_final += 1
                cv2.imwrite(str(no_det_cls / img_path.name), img)

        per_class_stats[cls_name] = {
            "总计": len(img_paths),
            "裁剪成功": c_crop,
            "整图保留": c_fullimg,
            "v11补检成功": c_v11_ok,
            "双模未检出": c_no_det_final,
            "跳过": c_skip,
        }
        logger.info(
            "[%s]  总计 %d | 裁剪 %d | 整图保留 %d | v11补检成功 %d | 双模未检出 %d | 跳过 %d",
            cls_name, len(img_paths), c_crop, c_fullimg, c_v11_ok, c_no_det_final, c_skip,
        )

    logger.info("=" * 60)
    logger.info("推理完成！主输出：%s", OUT_DIR)
    logger.info("双模均未检出样本仅写入（不在主目录）：%s", NO_DET_DIR)
    valid = total - skipped
    logger.info(
        "总图片: %d | 裁剪成功: %d | 整图保留: %d | 首模未检出后v11补检成功: %d | 双模均未检出: %d | 跳过: %d",
        total, cropped, fullimg, v11_recovered, no_detection_final, skipped,
    )
    logger.info(
        "说明：仅采纳狗类(id=16)；置信度≥%.2f；双模均未检出 %d 张（占有效图 %.1f%%）",
        CONF_THRESHOLD,
        no_detection_final,
        100.0 * no_detection_final / valid if valid else 0.0,
    )
    logger.info("各类别明细：")
    for cls_name in EMOTION_SUBDIRS:
        stat = per_class_stats[cls_name]
        logger.info(
            "  %-10s  总计 %d | 裁剪 %d | 整图 %d | v11补检 %d | 双模未检出 %d | 跳过 %d",
            cls_name,
            stat["总计"],
            stat["裁剪成功"],
            stat["整图保留"],
            stat["v11补检成功"],
            stat["双模未检出"],
            stat["跳过"],
        )
    logger.info("日志已保存：%s", log_path)


if __name__ == "__main__":
    main()
