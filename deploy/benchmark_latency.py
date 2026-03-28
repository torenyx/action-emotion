# -*- coding: utf-8 -*-
"""
宠物多模态情绪识别系统 —— 推理时延基准测试

测试内容：
  - 三种推理模式：仅图像 / 仅音频 / 图像+音频融合
  - 两种设备：CPU / CUDA（若可用）
  - 各阶段细粒度分解：预处理 · 模型前向 · 后处理(融合)
  - 统计指标：均值 · 标准差 · 中位数 · P95 · P99 · 最小值 · 最大值
  - 吞吐量（QPS）估算

运行：
  conda activate d2l
  python deploy/benchmark_latency.py
  python deploy/benchmark_latency.py --warmup 20 --repeats 200 --device cpu
  python deploy/benchmark_latency.py --real-image <路径> --real-audio <路径>
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TXT_DIR = ROOT / "txt"
TXT_DIR.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d%H%M%S")
_LOG_PATH = TXT_DIR / f"benchmark_latency_{TS}.txt"

# ════════════════════════════════════════════════════════════════
# 日志（同时写文件和终端）
# ════════════════════════════════════════════════════════════════

_log_lines: list[str] = []


def _log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(msg)


def _flush_log() -> None:
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))
    _log(f"\n[已保存] 测试报告 → {_LOG_PATH}")


# ════════════════════════════════════════════════════════════════
# 合成测试数据（无需真实文件）
# ════════════════════════════════════════════════════════════════

def _make_synthetic_image(path: Path) -> None:
    """生成一张 224×224 的随机 RGB PNG，写到临时路径。"""
    from PIL import Image as PILImage
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(str(path), format="PNG")


def _make_synthetic_audio(path: Path, duration_s: float = 2.0, sr: int = 22050) -> None:
    """生成一段随机白噪声 WAV，写到临时路径。"""
    import scipy.io.wavfile as wavfile
    n = int(sr * duration_s)
    samples = (np.random.randn(n) * 0.1 * 32767).astype(np.int16)
    wavfile.write(str(path), sr, samples)


# ════════════════════════════════════════════════════════════════
# 高精度计时工具
# ════════════════════════════════════════════════════════════════

def _sync_cuda(device: torch.device) -> None:
    """CUDA 模式下同步，确保计时准确。"""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timer_ns() -> int:
    return time.perf_counter_ns()


# ════════════════════════════════════════════════════════════════
# 统计汇总
# ════════════════════════════════════════════════════════════════

def _stats(times_ms: list[float]) -> dict[str, float]:
    a = np.array(times_ms, dtype=np.float64)
    return {
        "均值(ms)":   float(np.mean(a)),
        "标准差(ms)": float(np.std(a)),
        "中位数(ms)": float(np.median(a)),
        "P95(ms)":    float(np.percentile(a, 95)),
        "P99(ms)":    float(np.percentile(a, 99)),
        "最小(ms)":   float(np.min(a)),
        "最大(ms)":   float(np.max(a)),
        "QPS":        float(1000.0 / np.mean(a)),
    }


def _print_stats(label: str, st: dict[str, float]) -> None:
    _log(f"  [{label}]")
    _log(f"    均值={st['均值(ms)']:.2f}ms  标准差={st['标准差(ms)']:.2f}ms  "
         f"中位数={st['中位数(ms)']:.2f}ms")
    _log(f"    P95={st['P95(ms)']:.2f}ms  P99={st['P99(ms)']:.2f}ms  "
         f"最小={st['最小(ms)']:.2f}ms  最大={st['最大(ms)']:.2f}ms")
    _log(f"    吞吐量 ≈ {st['QPS']:.1f} QPS")


# ════════════════════════════════════════════════════════════════
# 细粒度分解计时：预处理 / 前向 / 后处理
# ════════════════════════════════════════════════════════════════

def _bench_image_only(
    system,
    img_path: Path,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> None:
    """仅图像模式细粒度时延分解。"""
    from deploy.try1 import load_image, fuse

    _log("  ── 细粒度分解（仅图像）")

    times_preprocess: list[float] = []
    times_forward: list[float] = []
    times_fuse: list[float] = []
    times_total: list[float] = []

    transform = system.img_transform

    for i in range(warmup + repeats):
        # ① 预处理
        _sync_cuda(device)
        t0 = _timer_ns()
        pixel_values = load_image(img_path, transform).to(device)
        _sync_cuda(device)
        t1 = _timer_ns()

        # ② 前向
        with torch.inference_mode():
            out_raw = system.img_model(pixel_values, species=None)
        _sync_cuda(device)
        t2 = _timer_ns()

        # ③ 后处理 / 融合
        img_logits = {k: v[0].cpu().float().numpy() for k, v in out_raw.items()}
        result = fuse(img_logits, None, None)
        t3 = _timer_ns()

        if i >= warmup:
            times_preprocess.append((t1 - t0) / 1e6)
            times_forward.append((t2 - t1) / 1e6)
            times_fuse.append((t3 - t2) / 1e6)
            times_total.append((t3 - t0) / 1e6)

    _print_stats("预处理", _stats(times_preprocess))
    _print_stats("模型前向", _stats(times_forward))
    _print_stats("融合/后处理", _stats(times_fuse))
    _print_stats("端到端合计", _stats(times_total))


def _bench_audio_only(
    system,
    audio_path: Path,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> None:
    """仅音频模式细粒度时延分解。"""
    from deploy.try1 import load_audio_mel, fuse

    _log("  ── 细粒度分解（仅音频）")

    times_preprocess: list[float] = []
    times_forward: list[float] = []
    times_fuse: list[float] = []
    times_total: list[float] = []

    for i in range(warmup + repeats):
        _sync_cuda(device)
        t0 = _timer_ns()
        mel = load_audio_mel(audio_path).to(device)
        _sync_cuda(device)
        t1 = _timer_ns()

        with torch.inference_mode():
            out_raw = system.audio_model(mel, species=None)
        _sync_cuda(device)
        t2 = _timer_ns()

        audio_logits = {k: v[0].cpu().float().numpy() for k, v in out_raw.items()}
        result = fuse(None, audio_logits, None)
        t3 = _timer_ns()

        if i >= warmup:
            times_preprocess.append((t1 - t0) / 1e6)
            times_forward.append((t2 - t1) / 1e6)
            times_fuse.append((t3 - t2) / 1e6)
            times_total.append((t3 - t0) / 1e6)

    _print_stats("预处理", _stats(times_preprocess))
    _print_stats("模型前向", _stats(times_forward))
    _print_stats("融合/后处理", _stats(times_fuse))
    _print_stats("端到端合计", _stats(times_total))


def _bench_multimodal(
    system,
    img_path: Path,
    audio_path: Path,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> None:
    """双模态融合细粒度时延分解。"""
    from deploy.try1 import load_image, load_audio_mel, fuse

    _log("  ── 细粒度分解（图像+音频融合）")

    times_img_pre: list[float] = []
    times_img_fwd: list[float] = []
    times_audio_pre: list[float] = []
    times_audio_fwd: list[float] = []
    times_fuse: list[float] = []
    times_total: list[float] = []

    transform = system.img_transform

    for i in range(warmup + repeats):
        _sync_cuda(device)
        t0 = _timer_ns()

        # 图像预处理
        pixel_values = load_image(img_path, transform).to(device)
        _sync_cuda(device)
        t1 = _timer_ns()

        # 图像前向
        with torch.inference_mode():
            out_img = system.img_model(pixel_values, species=None)
        _sync_cuda(device)
        t2 = _timer_ns()

        # 音频预处理
        mel = load_audio_mel(audio_path).to(device)
        _sync_cuda(device)
        t3 = _timer_ns()

        # 音频前向
        with torch.inference_mode():
            out_audio = system.audio_model(mel, species=None)
        _sync_cuda(device)
        t4 = _timer_ns()

        # 融合
        img_logits = {k: v[0].cpu().float().numpy() for k, v in out_img.items()}
        audio_logits = {k: v[0].cpu().float().numpy() for k, v in out_audio.items()}
        result = fuse(img_logits, audio_logits, None)
        t5 = _timer_ns()

        if i >= warmup:
            times_img_pre.append((t1 - t0) / 1e6)
            times_img_fwd.append((t2 - t1) / 1e6)
            times_audio_pre.append((t3 - t2) / 1e6)
            times_audio_fwd.append((t4 - t3) / 1e6)
            times_fuse.append((t5 - t4) / 1e6)
            times_total.append((t5 - t0) / 1e6)

    _print_stats("图像预处理", _stats(times_img_pre))
    _print_stats("图像前向",   _stats(times_img_fwd))
    _print_stats("音频预处理", _stats(times_audio_pre))
    _print_stats("音频前向",   _stats(times_audio_fwd))
    _print_stats("融合/后处理", _stats(times_fuse))
    _print_stats("端到端合计", _stats(times_total))


# ════════════════════════════════════════════════════════════════
# 主测试流程
# ════════════════════════════════════════════════════════════════

def _run_benchmark(
    device_str: str,
    img_path: Path,
    audio_path: Path,
    warmup: int,
    repeats: int,
) -> None:
    from deploy.try1 import MultimodalPetEmotionSystem

    device = torch.device(device_str)
    _log(f"\n{'═'*62}")
    _log(f"  设备: {device_str.upper()}  |  Warm-up={warmup}  Repeats={repeats}")
    _log(f"{'═'*62}")

    # 显示 CUDA 设备信息
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        _log(f"  GPU: {gpu_name}  显存: {total_mem:.1f} GB")

    # 加载模型（计时）
    _log("\n[模型加载]")
    t_load_start = time.perf_counter()
    system = MultimodalPetEmotionSystem(device=device_str)
    t_load_end = time.perf_counter()
    _log(f"  加载耗时: {(t_load_end - t_load_start)*1000:.1f} ms")

    # 显存占用（CUDA 模式）
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        _log(f"  显存已分配: {allocated:.1f} MB  已预留: {reserved:.1f} MB")

    # ── 模式 1：仅图像
    _log("\n[模式 1 / 3]  仅图像推理")
    _bench_image_only(system, img_path, device, warmup, repeats)

    # ── 模式 2：仅音频
    _log("\n[模式 2 / 3]  仅音频推理")
    _bench_audio_only(system, audio_path, device, warmup, repeats)

    # ── 模式 3：双模态融合
    _log("\n[模式 3 / 3]  图像+音频融合推理")
    _bench_multimodal(system, img_path, audio_path, device, warmup, repeats)

    # 内存统计（CPU 侧）
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / 1024**2
        _log(f"\n  进程内存占用 (RSS): {rss_mb:.1f} MB")
    except ImportError:
        pass

    _log(f"\n{'─'*62}")


# ════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="宠物多模态情绪识别 推理时延基准测试")
    parser.add_argument("--warmup",      type=int, default=10, help="预热次数（默认10）")
    parser.add_argument("--repeats",     type=int, default=100, help="正式重复次数（默认100）")
    parser.add_argument("--device",      type=str, default="auto",
                        choices=["auto", "cpu", "cuda"], help="测试设备（默认auto=优先cuda）")
    parser.add_argument("--real-image",  type=str, default=None, help="使用真实图像文件路径")
    parser.add_argument("--real-audio",  type=str, default=None, help="使用真实音频文件路径")
    args = parser.parse_args()

    _log("=" * 62)
    _log("  宠物多模态情绪识别系统 — 推理时延基准测试")
    _log(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("=" * 62)

    # 准备测试文件
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if args.real_image:
            img_path = Path(args.real_image)
            _log(f"  图像来源: 真实文件 ({img_path.name})")
        else:
            img_path = tmp / "synth_test.png"
            _make_synthetic_image(img_path)
            _log("  图像来源: 合成随机图像 (224×224 随机RGB PNG)")

        if args.real_audio:
            audio_path = Path(args.real_audio)
            _log(f"  音频来源: 真实文件 ({audio_path.name})")
        else:
            audio_path = tmp / "synth_test.wav"
            _make_synthetic_audio(audio_path, duration_s=2.0, sr=22050)
            _log("  音频来源: 合成随机音频 (2s 白噪声 22050Hz WAV)")

        _log(f"  Warm-up 次数: {args.warmup}")
        _log(f"  正式重复次数: {args.repeats}")

        # 确定测试设备列表
        if args.device == "auto":
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices = ["cuda", "cpu"]
            else:
                _log("\n  [提示] 未检测到 CUDA 设备，仅测试 CPU。")
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                _log("\n  [错误] 指定了 --device cuda 但未检测到可用 GPU，退出。")
                sys.exit(1)
            devices = ["cuda"]
        else:
            devices = ["cpu"]

        for dev in devices:
            _run_benchmark(dev, img_path, audio_path, args.warmup, args.repeats)

        _log("\n" + "=" * 62)
        _log("  基准测试完成")
        _log("=" * 62)
        _flush_log()


if __name__ == "__main__":
    main()
