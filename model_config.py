"""Configurable model selection for SWE-RL training.

Supports multiple coding models with automatic dtype and quantization
selection based on available hardware.  Designed for Phase 1:
local dev (GTX 1660 Ti 6GB) + Kaggle free (T4 16GB).

Usage::

    cfg = get_model_config("qwen-0.5b")
    model, tok = load_model(cfg, device="cuda")

    # Or auto-detect best config for hardware
    cfg = auto_select_model()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for one model."""

    name: str
    hf_id: str
    params: str  # human-readable, e.g. "0.5B"
    vram_fp16_gb: float  # approximate VRAM needed in fp16
    vram_4bit_gb: float  # approximate VRAM needed in 4-bit
    default_max_new_tokens: int = 200
    supports_4bit: bool = True
    context_length: int = 4096
    tags: tuple[str, ...] = ()


# ── model presets ───────────────────────────────────────────────────

QWEN_05B = ModelConfig(
    name="qwen-0.5b",
    hf_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    params="0.5B",
    vram_fp16_gb=1.0,
    vram_4bit_gb=0.5,
    context_length=4096,
    tags=("small", "fast", "pipeline-testing"),
)

QWEN_1_5B = ModelConfig(
    name="qwen-1.5b",
    hf_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    params="1.5B",
    vram_fp16_gb=3.0,
    vram_4bit_gb=1.5,
    context_length=4096,
    tags=("small", "local-dev"),
)

QWEN_3B = ModelConfig(
    name="qwen-3b",
    hf_id="Qwen/Qwen2.5-Coder-3B-Instruct",
    params="3B",
    vram_fp16_gb=6.0,
    vram_4bit_gb=2.5,
    context_length=4096,
    tags=("medium", "local-dev", "kaggle"),
)

QWEN_7B = ModelConfig(
    name="qwen-7b",
    hf_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    params="7B",
    vram_fp16_gb=14.0,
    vram_4bit_gb=5.0,
    context_length=8192,
    default_max_new_tokens=400,
    tags=("large", "kaggle", "quality"),
)

DEEPSEEK_1_3B = ModelConfig(
    name="deepseek-1.3b",
    hf_id="deepseek-ai/deepseek-coder-1.3b-instruct",
    params="1.3B",
    vram_fp16_gb=2.6,
    vram_4bit_gb=1.2,
    context_length=4096,
    tags=("small", "local-dev"),
)

DEEPSEEK_6_7B = ModelConfig(
    name="deepseek-6.7b",
    hf_id="deepseek-ai/deepseek-coder-6.7b-instruct",
    params="6.7B",
    vram_fp16_gb=13.4,
    vram_4bit_gb=4.5,
    context_length=8192,
    default_max_new_tokens=400,
    tags=("large", "kaggle", "quality"),
)

CODELLAMA_7B = ModelConfig(
    name="codellama-7b",
    hf_id="codellama/CodeLlama-7b-Instruct-hf",
    params="7B",
    vram_fp16_gb=14.0,
    vram_4bit_gb=5.0,
    context_length=4096,
    default_max_new_tokens=400,
    tags=("large", "kaggle"),
)

STARCODER2_3B = ModelConfig(
    name="starcoder2-3b",
    hf_id="bigcode/starcoder2-3b",
    params="3B",
    vram_fp16_gb=6.0,
    vram_4bit_gb=2.5,
    context_length=4096,
    tags=("medium", "local-dev", "kaggle"),
)

_REGISTRY: Mapping[str, ModelConfig] = {
    "qwen-0.5b": QWEN_05B,
    "qwen-1.5b": QWEN_1_5B,
    "qwen-3b": QWEN_3B,
    "qwen-7b": QWEN_7B,
    "deepseek-1.3b": DEEPSEEK_1_3B,
    "deepseek-6.7b": DEEPSEEK_6_7B,
    "codellama-7b": CODELLAMA_7B,
    "starcoder2-3b": STARCODER2_3B,
}


def get_model_config(name: str) -> ModelConfig:
    """Look up a model config by preset name.

    Raises ``KeyError`` with list of known models when *name* is unknown.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = sorted(_REGISTRY.keys())
        raise KeyError(f"Unknown model {name!r}. Known: {known}")
    return _REGISTRY[key]


def list_models() -> Sequence[str]:
    """Return all preset model names."""
    return sorted(_REGISTRY.keys())


def models_for_vram(vram_gb: float, *, quantize: bool = False) -> list[ModelConfig]:
    """Return models that fit in the given VRAM budget, largest first."""
    fits = []
    for cfg in _REGISTRY.values():
        needed = cfg.vram_4bit_gb if quantize else cfg.vram_fp16_gb
        if needed <= vram_gb:
            fits.append(cfg)
    # Deduplicate (aliases) and sort by VRAM desc (largest fitting first)
    seen = set()
    unique = []
    for c in fits:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    key = (lambda c: c.vram_4bit_gb) if quantize else (lambda c: c.vram_fp16_gb)
    return sorted(unique, key=key, reverse=True)


# ── hardware detection ──────────────────────────────────────────────

@dataclass(frozen=True)
class HardwareInfo:
    """Detected hardware capabilities."""

    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    ram_free_gb: float
    cpu_count: int
    runtime: str  # "local", "kaggle", "colab", "unknown"


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and runtime environment."""
    runtime = _detect_runtime()
    gpu_name, vram_total, vram_free = _detect_gpu()
    ram_total, ram_free = _detect_ram()
    cpu_count = os.cpu_count() or 1

    return HardwareInfo(
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        ram_total_gb=ram_total,
        ram_free_gb=ram_free,
        cpu_count=cpu_count,
        runtime=runtime,
    )


def auto_select_model(hw: HardwareInfo | None = None) -> tuple[ModelConfig, bool]:
    """Pick the best model for detected hardware.

    Returns ``(config, use_4bit)``.  Prefers the largest model that fits
    in available VRAM, using 4-bit quantization when needed.
    """
    if hw is None:
        hw = detect_hardware()

    vram = hw.vram_free_gb if hw.vram_free_gb > 0 else hw.vram_total_gb

    # Try fp16 first
    fp16_fits = models_for_vram(vram, quantize=False)
    if fp16_fits:
        return fp16_fits[0], False

    # Fall back to 4-bit
    q4_fits = models_for_vram(vram, quantize=True)
    if q4_fits:
        return q4_fits[0], True

    # Nothing fits — return smallest model in fp16
    return QWEN_05B, False


def print_hardware_summary(hw: HardwareInfo) -> None:
    """Print a human-readable hardware summary."""
    print(f"Runtime:  {hw.runtime}")
    print(f"GPU:      {hw.gpu_name} ({hw.vram_total_gb:.1f} GB total, "
          f"{hw.vram_free_gb:.1f} GB free)")
    print(f"RAM:      {hw.ram_total_gb:.1f} GB total, {hw.ram_free_gb:.1f} GB free")
    print(f"CPUs:     {hw.cpu_count}")


def recommend_workers(hw: HardwareInfo, model_cfg: ModelConfig,
                      use_4bit: bool = False) -> int:
    """Suggest max parallel workers based on available RAM."""
    model_ram = model_cfg.vram_4bit_gb if use_4bit else model_cfg.vram_fp16_gb
    # Reserve RAM for model + 2GB headroom
    available = hw.ram_free_gb - model_ram - 2.0
    # Each Docker/subprocess worker needs ~1.5GB for Rust, ~1GB for C#
    per_worker = 1.5
    workers = max(1, int(available / per_worker))
    return min(workers, hw.cpu_count // 2, 8)  # cap at half CPUs or 8


# ── internals ───────────────────────────────────────────────────────

def _detect_runtime() -> str:
    """Detect if running on Kaggle, Colab, or local."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    return "local"


def _detect_gpu() -> tuple[str, float, float]:
    """Return (name, total_gb, free_gb).  Falls back to (none, 0, 0)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            free = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
            return name, round(total, 1), round(free, 1)
    except Exception:
        pass
    return "none", 0.0, 0.0


def _detect_ram() -> tuple[float, float]:
    """Return (total_gb, available_gb)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / (1024 ** 3), 1), round(vm.available / (1024 ** 3), 1)
    except ImportError:
        pass
    # Fallback: read /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) / (1024 * 1024)
            total = info.get("MemTotal", 0.0)
            avail = info.get("MemAvailable", info.get("MemFree", 0.0))
            return round(total, 1), round(avail, 1)
    except Exception:
        return 0.0, 0.0
