from __future__ import annotations

import logging
import os
import platform
import tempfile
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

RequestedDevice = Literal["auto", "apple_gpu", "cpu"]
ResolvedDevice = Literal["apple_gpu", "cpu"]


@dataclass
class EngineDebugInfo:
    requested: RequestedDevice
    resolved: ResolvedDevice
    backend: str
    reason: str


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration: Optional[float]
    segments: List[SegmentResult]
    debug: Optional[dict[str, str]] = None


class GpuNotAvailableError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"GPU not available: {reason}")


_MODEL_CACHE: dict[str, WhisperModel] = {}


def _resolve_device() -> RequestedDevice:
    raw = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    mapping = {
        "intel_gpu": "apple_gpu",
        "apple_gpu": "apple_gpu",
        "mac_gpu": "apple_gpu",
    }
    raw = mapping.get(raw, raw)
    if raw in {"auto", "apple_gpu", "cpu"}:
        return raw
    logger.warning("Unknown WHISPER_DEVICE=%s, fallback to auto", raw)
    return "auto"


def _is_macos_arm64() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _probe_mlx_gpu() -> tuple[bool, str]:
    if not _is_macos_arm64():
        return False, "platform_not_macos_arm64"

    try:
        import mlx.core as mx
    except Exception as exc:
        return False, f"mlx_unavailable:{exc.__class__.__name__}"

    try:
        default_device = str(mx.default_device())
        if "gpu" in default_device.lower():
            return True, f"mlx_gpu_available:{default_device}"
        return False, f"mlx_gpu_not_selected:{default_device}"
    except Exception as exc:
        return False, f"mlx_probe_failed:{exc.__class__.__name__}:{exc}"


def check_gpu_support() -> tuple[bool, str]:
    return _probe_mlx_gpu()


def _resolve_engine(requested: RequestedDevice) -> EngineDebugInfo:
    if requested == "cpu":
        return EngineDebugInfo(requested=requested, resolved="cpu", backend="ctranslate2", reason="forced_cpu")

    available, reason = _probe_mlx_gpu()
    if available:
        return EngineDebugInfo(requested=requested, resolved="apple_gpu", backend="mlx-whisper", reason=reason)

    return EngineDebugInfo(requested=requested, resolved="cpu", backend="ctranslate2", reason=reason)


def _model_cache_key(model_name: str) -> str:
    return f"{model_name}|cpu|int8"


def _get_cpu_model(model_name: str) -> WhisperModel:
    cache_key = _model_cache_key(model_name)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _MODEL_CACHE[cache_key]


def _transcribe_with_mlx(
    temp_path: str,
    model_name: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
) -> tuple[str, Optional[str], Optional[float], List[SegmentResult]]:
    import mlx_whisper

    result: Any = mlx_whisper.transcribe(
        temp_path,
        path_or_hf_repo=model_name,
        language=language,
        initial_prompt=prompt,
        temperature=temperature,
    )
    text = str(result.get("text", "")).strip()
    language_out = result.get("language")

    segments: List[SegmentResult] = []
    for idx, seg in enumerate(result.get("segments", []) or []):
        cleaned = str(seg.get("text", "")).strip()
        segments.append(
            SegmentResult(
                id=idx,
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=cleaned,
            )
        )

    duration = None
    if segments:
        duration = max(s.end for s in segments)

    return text, language_out, duration, segments


def transcribe_audio(
    audio_bytes: bytes,
    model_name: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    include_debug: bool = False,
    require_gpu: bool = False,
) -> TranscriptionResult:
    requested = _resolve_device()
    engine = _resolve_engine(requested)
    if require_gpu and engine.resolved != "apple_gpu":
        raise GpuNotAvailableError(engine.reason)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
        if engine.resolved == "apple_gpu":
            try:
                text, language_out, duration, segments = _transcribe_with_mlx(
                    temp_path=temp_path,
                    model_name=model_name,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                )
            except Exception as exc:
                engine = EngineDebugInfo(
                    requested=engine.requested,
                    resolved="cpu",
                    backend="ctranslate2",
                    reason=f"apple_gpu_fallback_to_cpu:{exc.__class__.__name__}",
                )
                model = _get_cpu_model(model_name)
                raw_segments, info = model.transcribe(
                    temp_path,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                )
                segments = []
                texts: List[str] = []
                for idx, seg in enumerate(raw_segments):
                    cleaned = seg.text.strip()
                    if cleaned:
                        texts.append(cleaned)
                    segments.append(SegmentResult(id=idx, start=float(seg.start), end=float(seg.end), text=cleaned))
                text = " ".join(texts).strip()
                language_out = getattr(info, "language", language)
                duration = getattr(info, "duration", None)
        else:
            model = _get_cpu_model(model_name)
            raw_segments, info = model.transcribe(
                temp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )
            segments = []
            texts = []
            for idx, seg in enumerate(raw_segments):
                cleaned = seg.text.strip()
                if cleaned:
                    texts.append(cleaned)
                segments.append(SegmentResult(id=idx, start=float(seg.start), end=float(seg.end), text=cleaned))
            text = " ".join(texts).strip()
            language_out = getattr(info, "language", language)
            duration = getattr(info, "duration", None)

        return TranscriptionResult(
            text=text,
            language=language_out,
            duration=duration,
            segments=segments,
            debug=engine.__dict__ if include_debug else None,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
