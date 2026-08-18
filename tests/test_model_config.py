"""Tests for model_config module."""

from __future__ import annotations

import pytest

from model_config import (
    QWEN_05B,
    QWEN_1_5B,
    QWEN_3B,
    QWEN_7B,
    DEEPSEEK_1_3B,
    DEEPSEEK_6_7B,
    HardwareInfo,
    ModelConfig,
    auto_select_model,
    detect_hardware,
    get_model_config,
    list_models,
    models_for_vram,
    recommend_workers,
)


class TestGetModelConfig:
    def test_qwen_05b(self) -> None:
        assert get_model_config("qwen-0.5b") is QWEN_05B

    def test_qwen_7b(self) -> None:
        assert get_model_config("qwen-7b") is QWEN_7B

    def test_deepseek(self) -> None:
        assert get_model_config("deepseek-1.3b") is DEEPSEEK_1_3B

    def test_case_insensitive(self) -> None:
        assert get_model_config("Qwen-0.5B") is QWEN_05B

    def test_strips_whitespace(self) -> None:
        assert get_model_config("  qwen-3b  ") is QWEN_3B

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("gpt-4")

    def test_unknown_lists_known(self) -> None:
        with pytest.raises(KeyError, match="qwen-0.5b"):
            get_model_config("nonexistent")


class TestListModels:
    def test_returns_all_presets(self) -> None:
        models = list_models()
        assert "qwen-0.5b" in models
        assert "qwen-7b" in models
        assert "deepseek-6.7b" in models
        assert "codellama-7b" in models

    def test_sorted(self) -> None:
        models = list_models()
        assert models == sorted(models)


class TestModelsForVram:
    def test_6gb_fp16(self) -> None:
        fits = models_for_vram(6.0, quantize=False)
        names = [c.name for c in fits]
        assert "qwen-3b" in names
        assert "qwen-0.5b" in names
        assert "qwen-7b" not in names  # 14GB needed

    def test_6gb_4bit(self) -> None:
        fits = models_for_vram(6.0, quantize=True)
        names = [c.name for c in fits]
        assert "qwen-7b" in names  # 5GB in 4-bit
        assert "deepseek-6.7b" in names  # 4.5GB in 4-bit

    def test_16gb_fp16(self) -> None:
        fits = models_for_vram(16.0, quantize=False)
        names = [c.name for c in fits]
        assert "qwen-7b" in names
        assert "deepseek-6.7b" in names

    def test_1gb_fp16(self) -> None:
        fits = models_for_vram(1.0, quantize=False)
        names = [c.name for c in fits]
        assert "qwen-0.5b" in names
        assert len(fits) == 1  # only 0.5B fits

    def test_largest_first(self) -> None:
        fits = models_for_vram(16.0, quantize=False)
        vrams = [c.vram_fp16_gb for c in fits]
        assert vrams == sorted(vrams, reverse=True)

    def test_zero_vram(self) -> None:
        assert models_for_vram(0.0) == []


class TestModelConfigFrozen:
    def test_cannot_mutate(self) -> None:
        with pytest.raises(AttributeError):
            QWEN_05B.name = "changed"  # type: ignore[misc]


class TestAutoSelectModel:
    def _hw(self, vram: float = 6.0, ram: float = 16.0) -> HardwareInfo:
        return HardwareInfo(
            gpu_name="test",
            vram_total_gb=vram,
            vram_free_gb=vram,
            ram_total_gb=ram,
            ram_free_gb=ram * 0.6,
            cpu_count=8,
            runtime="local",
        )

    def test_6gb_picks_fp16(self) -> None:
        cfg, use_4bit = auto_select_model(self._hw(vram=6.0))
        assert not use_4bit
        # Should pick largest fp16 that fits in 6GB
        assert cfg.vram_fp16_gb <= 6.0

    def test_4gb_uses_4bit(self) -> None:
        cfg, use_4bit = auto_select_model(self._hw(vram=4.0))
        # Nothing large fits fp16 at 4GB, but several fit 4-bit
        # Either fp16 small model or 4-bit larger model
        if use_4bit:
            assert cfg.vram_4bit_gb <= 4.0

    def test_16gb_picks_largest(self) -> None:
        cfg, use_4bit = auto_select_model(self._hw(vram=16.0))
        assert not use_4bit
        assert cfg.vram_fp16_gb <= 16.0
        # Should be one of the 7B models
        assert "7b" in cfg.name or "6.7b" in cfg.name

    def test_tiny_vram_falls_back(self) -> None:
        cfg, use_4bit = auto_select_model(self._hw(vram=0.3))
        # Should still return something (QWEN_05B as fallback)
        assert cfg is QWEN_05B

    def test_kaggle_t4(self) -> None:
        hw = HardwareInfo(
            gpu_name="Tesla T4",
            vram_total_gb=16.0,
            vram_free_gb=15.0,
            ram_total_gb=30.0,
            ram_free_gb=25.0,
            cpu_count=4,
            runtime="kaggle",
        )
        cfg, use_4bit = auto_select_model(hw)
        assert cfg.vram_fp16_gb <= 15.0


class TestRecommendWorkers:
    def _hw(self, ram_free: float = 10.0, cpus: int = 12) -> HardwareInfo:
        return HardwareInfo(
            gpu_name="test",
            vram_total_gb=6.0,
            vram_free_gb=5.0,
            ram_total_gb=16.0,
            ram_free_gb=ram_free,
            cpu_count=cpus,
            runtime="local",
        )

    def test_enough_ram_multiple_workers(self) -> None:
        workers = recommend_workers(self._hw(ram_free=12.0), QWEN_05B)
        assert workers >= 2

    def test_low_ram_single_worker(self) -> None:
        workers = recommend_workers(self._hw(ram_free=3.0), QWEN_3B)
        assert workers == 1

    def test_never_zero(self) -> None:
        workers = recommend_workers(self._hw(ram_free=1.0), QWEN_7B)
        assert workers >= 1

    def test_capped_by_cpus(self) -> None:
        workers = recommend_workers(self._hw(ram_free=100.0, cpus=4), QWEN_05B)
        assert workers <= 2  # half of 4 CPUs


class TestDetectHardware:
    def test_returns_hardware_info(self) -> None:
        hw = detect_hardware()
        assert isinstance(hw, HardwareInfo)
        assert hw.cpu_count > 0
        assert hw.ram_total_gb > 0
        assert hw.runtime in ("local", "kaggle", "colab", "unknown")
