# -*- coding: utf-8 -*-
"""
CED-Mini 音频单模态猫狗情绪识别 —— 主训练入口

架构：
  CED-Mini (ViT-256d-12L, 9.6M) 编码器
    └── 冻结底层 8 层，解冻顶层 4 层 + BN/LN
    └── 差异化学习率：骨干 2e-5, 分类头 5e-4
  共享投影 (256→512) + ResBlock×2
  物种二分类头 head_species (2 类) + 情绪头（与 train/3 一致：训练用 GT 路由，测试 species=None 端到端）
    - dog_audio: 4 类 (barking / growling / howling / whining)
    - cat_audio: 10 类 (Angry / Defence / ... / Warning)

训练策略（与 train/3_dinov3_convnext_finetune 对齐思路）：
  - 仅 WeightedRandomSampler 做逆频均衡；Focal 不再叠加逆频 class weight；狗/猫两任务损失等权 1.0
  - 训练期对狗音频行施加轻微 Mel 高斯噪声（样本常少于猫音频，对标少数侧更强增强）
  - 特征空间 Mixup (alpha=0.3)
  - Warmup + CosineAnnealing
  - 早停与最优权重：验证 balanced_macro_F1（两任务 macro-F1 平均），不用准确率
  - 固定训练/验证划分 + TTA（噪声 / 时间平移 / 频域与时间轻量掩码轮换）
  - SpecAugment（训练期）与 EMA 最优权重（验证与保存）
  - AMP 混合精度

运行：
  conda activate d2l
  python cedtrain/run_train.py

输出：
  moxing/CedMini_AudioEmotion_{时间戳}.pkl
  figure/ced_train_curve_{时间戳}.png
  figure/ced_confusion_matrix_{时间戳}.png
  figure/ced_interpret_prf_{时间戳}.png、ced_interpret_roc_{时间戳}.png、
    ced_interpret_confidence_{时间戳}.png、ced_interpret_top_confusion_{时间戳}.png（可解释性）
  figure/ced_tsne_{时间戳}.png（共享表征 t-SNE）
  figure/ced_mel_gradcam_{时间戳}.png（Mel 频谱 Token-wise GradCAM）
  txt/ced_train_log_{时间戳}.txt
"""

import sys
from pathlib import Path

# 将项目根目录加入搜索路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from cedtrain.config import TrainConfig, MODEL_DIR, FIG_DIR, TASK_META
from cedtrain.utils import seed_everything, setup_logger, log_environment, get_timestamp
from cedtrain.data import (
    AudioEmotionDataset,
    stratified_split,
    make_weighted_sampler,
    make_collate_fn,
)
from cedtrain.modeling import CedAudioEmotionModel
from cedtrain.engine import (
    train_fold,
    full_evaluation,
    ensemble_evaluation,
)
from cedtrain.visualization import (
    plot_training_curves,
    plot_confusion_matrices,
)


def main():
    cfg = TrainConfig()
    ts = get_timestamp()

    logger, log_path = setup_logger("cedtrain", prefix="ced_train")

    logger.info("=" * 60)
    logger.info("CED-Mini 音频单模态猫狗情绪识别 正式训练")
    logger.info("=" * 60)
    log_environment(logger)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"计算设备: {device}")

    logger.info(
        f"超参数: epochs={cfg.num_epochs}, batch={cfg.batch_size}, "
        f"backbone_lr={cfg.backbone_lr}, head_lr={cfg.head_lr}, "
        f"patience={cfg.patience}, mixup_alpha={cfg.mixup_alpha}, "
        f"focal_gamma={cfg.focal_gamma}, dog_audio_mel_noise_std={cfg.dog_audio_mel_noise_std}, "
        f"species_loss_weight={cfg.species_loss_weight}, "
        f"spec_augment={cfg.spec_augment_enabled}, use_ema={cfg.use_ema}, ema_decay={cfg.ema_decay}, "
        f"tta_steps={cfg.tta_steps}, use_amp={cfg.use_amp}"
    )
    logger.info(
        "数据策略: 逆频采样器 + Focal(无 class weight) + 任务等权；勿与旧版逆频 loss 混用。",
    )
    logger.info("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────────────────
    logger.info("\n[1/5] 加载音频数据集...")
    dataset = AudioEmotionDataset(cfg)
    collate_fn = make_collate_fn(cfg)

    # ── 2. 分层划分 ──────────────────────────────────────────
    logger.info("\n[2/5] 分层采样划分数据集...")
    train_idx, val_idx, test_idx = stratified_split(
        dataset, cfg.train_ratio, cfg.val_ratio, cfg.seed,
    )

    logger.info(f"  训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}")
    logger.info("  训练模式: 固定分层划分（无 K 折交叉验证）")

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, persistent_workers=True,
    )

    # ── 3. K 折训练 ──────────────────────────────────────────
    logger.info("\n[3/5] 开始训练...")
    fold_states = []
    fold_histories = []

    loader_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    train_sampler = make_weighted_sampler(dataset, train_idx)
    train_loader = DataLoader(
        Subset(dataset, train_idx), sampler=train_sampler, **loader_kw,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), shuffle=False, **loader_kw,
    )

    state, history = train_fold(
        0, train_loader, val_loader,
        dataset, train_idx, device, cfg,
    )
    fold_states.append(state)
    fold_histories.append(history)

    # ── 4. 绘制训练曲线 ─────────────────────────────────────
    logger.info("\n[4/5] 绘制训练曲线...")
    fig_path = FIG_DIR / f"ced_train_curve_{ts}.png"
    plot_training_curves(fold_histories[-1], fig_path)
    logger.info(f"  训练曲线已保存: {fig_path}")

    # ── 4b. 最后一折模型在测试集上的单次评估（无 TTA，便于与集成对比）────
    logger.info("\n[4b] 最优权重 — 测试集单次评估（无 TTA）...")
    _eval_model = CedAudioEmotionModel(cfg).to(device)
    _eval_model.load_state_dict(fold_states[-1])
    full_evaluation(_eval_model, test_loader, device, dataset, cfg)
    del _eval_model

    # ── 5. 集成评估 ──────────────────────────────────────────
    logger.info(f"\n[5/5] 测试集评估（TTA {cfg.tta_steps} 步，单模型平均多视角）...")
    cm_data, prob_bundle = ensemble_evaluation(
        fold_states, test_loader, device, dataset, cfg,
    )
    if cm_data:
        cm_path = FIG_DIR / f"ced_confusion_matrix_{ts}.png"
        plot_confusion_matrices(cm_data, cm_path)
        logger.info(f"  混淆矩阵已保存: {cm_path}")
        try:
            from cedtrain.mlp_interpretability import (
                plot_tsne_ced_audio,
                run_mlp_interpretability_suite,
            )

            run_mlp_interpretability_suite(
                cm_data,
                prob_bundle,
                dataset.task_class_names,
                FIG_DIR,
                ts,
                prefix="ced_",
            )
            tsne_path = FIG_DIR / f"ced_tsne_{ts}.png"
            plot_tsne_ced_audio(
                fold_states[0],
                dataset,
                test_idx,
                device,
                cfg,
                tsne_path,
            )
            from cedtrain.mlp_interpretability import plot_mel_gradcam_samples
            gradcam_path = FIG_DIR / f"ced_mel_gradcam_{ts}.png"
            plot_mel_gradcam_samples(
                fold_states[0],
                dataset,
                test_idx,
                device,
                cfg,
                gradcam_path,
                n_samples_per_task=4,
            )
            logger.info("  可解释性组图（P/R/F1、ROC、置信度、Top 混淆）、t-SNE 与 Mel GradCAM 已写入 figure/")
        except Exception as e:
            logger.warning("可解释性补充图（P/R/F1、ROC、置信度、Top 混淆、t-SNE）失败：%s", e)

    # ── 保存模型 ─────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"CedMini_AudioEmotion_{ts}.pkl"
    torch.save({
        "model_states": fold_states,
        "n_folds": 1,
        "config": {
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "backbone_lr": cfg.backbone_lr,
            "head_lr": cfg.head_lr,
            "n_folds": 1,
            "tta_steps": cfg.tta_steps,
            "use_amp": cfg.use_amp,
            "mixup_alpha": cfg.mixup_alpha,
            "focal_gamma": cfg.focal_gamma,
            "dog_audio_mel_noise_std": cfg.dog_audio_mel_noise_std,
            "mel_top_db": cfg.mel_top_db,
            "spec_augment_enabled": cfg.spec_augment_enabled,
            "use_ema": cfg.use_ema,
            "ema_decay": cfg.ema_decay,
            "tta_time_shift_max": cfg.tta_time_shift_max,
            "tta_freq_mask_max": cfg.tta_freq_mask_max,
            "tta_time_mask_max": cfg.tta_time_mask_max,
            "species_loss_weight": cfg.species_loss_weight,
            "task_loss_equal": True,
            "hidden_dim": CedAudioEmotionModel.HIDDEN_DIM,
            "freeze_bottom_layers": CedAudioEmotionModel.FREEZE_BOTTOM_LAYERS,
        },
        "task_meta": {
            k: {"classes": v["classes"], "num_classes": v["num_classes"]}
            for k, v in TASK_META.items()
        },
    }, model_path)
    logger.info(f"  模型已保存 (单份最优权重): {model_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"训练日志: {log_path}")
    logger.info(f"模型文件: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
