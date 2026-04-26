"""Whisper transcription service with Intel GPU -> CPU fallback strategy.

Environment variables
---------------------
WHISPER_DEVICE: auto | intel_gpu | cpu
WHISPER_MODEL_SIZE: tiny | base | small | medium | large-v3 (and variants)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhisperSettings:
    """Runtime settings loaded from environment variables."""

    device_preference: str = "auto"
    model_size: str = "base"

    @classmethod
    def from_env(cls) -> "WhisperSettings":
        raw_device = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base").strip().lower()

        if raw_device not in {"auto", "intel_gpu", "cpu"}:
            raise ValueError(
                "WHISPER_DEVICE must be one of: auto, intel_gpu, cpu "
                f"(got: {raw_device!r})"
            )

        return cls(device_preference=raw_device, model_size=model_size)


@dataclass(frozen=True)
class DeviceSelection:
    """Resolved device/backend after startup probing."""

    requested: str
    resolved_device: str
    backend: str
    reason: str


class WhisperEngine:
    """Whisper inference wrapper with Intel GPU (OpenVINO) preferred."""

    def __init__(self, settings: WhisperSettings | None = None) -> None:
        self.settings = settings or WhisperSettings.from_env()
        self.selection = self._select_device()
        LOGGER.info(
            "Whisper startup: requested=%s, resolved=%s, backend=%s, model=%s, reason=%s",
            self.selection.requested,
            self.selection.resolved_device,
            self.selection.backend,
            self.settings.model_size,
            self.selection.reason,
        )

    def _select_device(self) -> DeviceSelection:
        pref = self.settings.device_preference

        if pref == "cpu":
            return DeviceSelection(
                requested=pref,
                resolved_device="cpu",
                backend="whisper",
                reason="WHISPER_DEVICE=cpu",
            )

        intel_gpu_ok, reason = _intel_gpu_available_for_openvino()
        if intel_gpu_ok and pref in {"auto", "intel_gpu"}:
            return DeviceSelection(
                requested=pref,
                resolved_device="intel_gpu",
                backend="openvino",
                reason=reason,
            )

        return DeviceSelection(
            requested=pref,
            resolved_device="cpu",
            backend="whisper",
            reason=f"fallback_to_cpu: {reason}",
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Transcribe audio and optionally return runtime debug fields.

        Note: this function intentionally focuses on runtime device strategy.
        Wire this to your actual model invocation in your API layer.
        """
        # Placeholder transcription for integration points.
        text = f"[demo] transcribed {audio_path} with {self.selection.backend}"
        response: dict[str, Any] = {
            "text": text,
            "model_size": self.settings.model_size,
        }

        if language:
            response["language"] = language

        if include_debug:
            response["debug"] = {
                "device": self.selection.resolved_device,
                "backend": self.selection.backend,
                "requested_device": self.selection.requested,
                "reason": self.selection.reason,
            }

        return response


def _intel_gpu_available_for_openvino() -> tuple[bool, str]:
    """Return True when OpenVINO Runtime can see Intel GPU device."""
    if importlib.util.find_spec("openvino.runtime") is None:
        return False, "openvino.runtime_not_installed"

    ov_runtime = importlib.import_module("openvino.runtime")

    try:
        core = ov_runtime.Core()
        devices = {d.upper() for d in core.available_devices}
    except Exception as exc:  # runtime/device probing exception only
        return False, f"openvino_probe_failed: {exc}"

    if "GPU" in devices:
        return True, "openvino_gpu_available"

    return False, f"openvino_devices={sorted(devices)}"


def create_engine() -> WhisperEngine:
    """Factory used by API startup hooks."""
    return WhisperEngine(WhisperSettings.from_env())
