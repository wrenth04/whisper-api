from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


RequestedDevice = Literal["auto", "intel_gpu", "cpu"]
ResolvedDevice = Literal["intel_gpu", "cpu"]


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


_MODEL_CACHE: dict[str, WhisperModel] = {}


def _resolve_device() -> RequestedDevice:
    raw = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    if raw in {"auto", "intel_gpu", "cpu"}:
        return raw
    logger.warning("Unknown WHISPER_DEVICE=%s, fallback to auto", raw)
    return "auto"


def _probe_openvino_gpu() -> tuple[bool, str]:
    try:
        from openvino.runtime import Core
    except Exception as exc:  # pragma: no cover - runtime dependent
        return False, f"openvino.runtime_unavailable:{exc.__class__.__name__}"

    try:
        devices = list(Core().available_devices)
    except Exception as exc:  # pragma: no cover - runtime dependent
        return False, f"openvino.probe_failed:{exc.__class__.__name__}:{exc}"

    if "GPU" in devices:
        return True, f"openvino_gpu_available:{devices}"
    return False, f"openvino_gpu_missing:{devices}"


def _resolve_engine(requested: RequestedDevice) -> EngineDebugInfo:
    if requested == "cpu":
        return EngineDebugInfo(
            requested=requested,
            resolved="cpu",
            backend="ctranslate2",
            reason="forced_cpu",
        )

    available, reason = _probe_openvino_gpu()
    if available:
        return EngineDebugInfo(
            requested=requested,
            resolved="intel_gpu",
            backend="openvino",
            reason=reason,
        )

    return EngineDebugInfo(
        requested=requested,
        resolved="cpu",
        backend="ctranslate2",
        reason=reason,
    )


def _model_runtime_args(engine: EngineDebugInfo) -> tuple[str, str]:
    if engine.resolved == "intel_gpu":
        return "auto", "int8_float16"
    return "cpu", "int8"


def _model_cache_key(model_name: str, device: str, compute_type: str) -> str:
    return f"{model_name}|{device}|{compute_type}"


def _get_model(model_name: str, engine: EngineDebugInfo) -> WhisperModel:
    device, compute_type = _model_runtime_args(engine)
    cache_key = _model_cache_key(model_name, device, compute_type)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    return _MODEL_CACHE[cache_key]


def _fallback_to_cpu(engine: EngineDebugInfo, reason: str) -> EngineDebugInfo:
    return EngineDebugInfo(
        requested=engine.requested,
        resolved="cpu",
        backend="ctranslate2",
        reason=f"gpu_fallback_to_cpu:{reason}",
    )


def transcribe_audio(
    audio_bytes: bytes,
    model_name: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    include_debug: bool = False,
) -> TranscriptionResult:
    requested = _resolve_device()
    engine = _resolve_engine(requested)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
        model = _get_model(model_name, engine)
        try:
            segments, info = model.transcribe(
                temp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )
        except Exception as exc:
            if engine.resolved != "intel_gpu":
                raise
            engine = _fallback_to_cpu(engine, f"transcribe_failed:{exc.__class__.__name__}")
            model = _get_model(model_name, engine)
            segments, info = model.transcribe(
                temp_path,
                language=language,
                initial_prompt=prompt,
                temperature=temperature,
            )

        segment_results: List[SegmentResult] = []
        texts: List[str] = []
        for idx, seg in enumerate(segments):
            cleaned = seg.text.strip()
            if cleaned:
                texts.append(cleaned)
            segment_results.append(
                SegmentResult(
                    id=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=cleaned,
                )
            )

        debug_payload = asdict(engine)
        logger.info("whisper_engine=%s", debug_payload)

        return TranscriptionResult(
            text=" ".join(texts).strip(),
            language=getattr(info, "language", language),
            duration=getattr(info, "duration", None),
            segments=segment_results,
            debug=debug_payload if include_debug else None,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
