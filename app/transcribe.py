from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
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


class GpuNotAvailableError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"GPU not available: {reason}")


_MODEL_CACHE: dict[str, WhisperModel] = {}
_MODEL_NAME_ALIASES: dict[str, str] = {
    "openvino/whisper-large-v3-int8-ov": "large-v3",
    "openvino/whisper-large-v3-fp16-ov": "large-v3",
    "openvino/whisper-small-int8-ov": "small",
    "openvino/whisper-base-int8-ov": "base",
    "openvino/whisper-tiny-int8-ov": "tiny",
}


def _resolve_device() -> RequestedDevice:
    raw = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
    if raw in {"auto", "intel_gpu", "cpu"}:
        return raw
    logger.warning("Unknown WHISPER_DEVICE=%s, fallback to auto", raw)
    return "auto"


def _cpu_threads_target() -> int:
    cpu_count = os.cpu_count() or 1
    raw_ratio = os.getenv("WHISPER_CPU_USAGE_RATIO", "0.8").strip()
    try:
        ratio = float(raw_ratio)
    except ValueError:
        logger.warning("Invalid WHISPER_CPU_USAGE_RATIO=%s, fallback to 0.8", raw_ratio)
        ratio = 0.8
    ratio = min(max(ratio, 0.1), 1.0)
    threads = max(1, math.floor(cpu_count * ratio))
    logger.info(
        "whisper_cpu_threads_config cpu_count=%s ratio=%.2f threads=%s",
        cpu_count,
        ratio,
        threads,
    )
    return threads


def check_gpu_support() -> tuple[bool, str]:
    return False, "gpu_probe_not_supported_without_openvino"


def _resolve_engine(requested: RequestedDevice) -> EngineDebugInfo:
    if requested == "cpu":
        return EngineDebugInfo(requested=requested, resolved="cpu", backend="ctranslate2", reason="forced_cpu")

    if requested == "intel_gpu":
        return EngineDebugInfo(
            requested=requested,
            resolved="intel_gpu",
            backend="ctranslate2",
            reason="forced_intel_gpu_no_preprobe",
        )

    return EngineDebugInfo(
        requested=requested,
        resolved="cpu",
        backend="ctranslate2",
        reason="auto_defaults_to_cpu",
    )


def _model_runtime_args(engine: EngineDebugInfo, compute_type: Optional[str] = None) -> tuple[str, str]:
    if engine.resolved == "intel_gpu":
        return "auto", compute_type or "int8_float16"
    return "cpu", "int8"


def _model_cache_key(model_name: str, device: str, compute_type: str, cpu_threads: int) -> str:
    return f"{model_name}|{device}|{compute_type}|cpu_threads={cpu_threads}"


def _get_model(model_name: str, engine: EngineDebugInfo, compute_type: Optional[str] = None) -> WhisperModel:
    device, compute_type = _model_runtime_args(engine, compute_type=compute_type)
    cpu_threads = _cpu_threads_target()
    cache_key = _model_cache_key(model_name, device, compute_type, cpu_threads)
    if cache_key not in _MODEL_CACHE:
        logger.info(
            "whisper_model_init cache_miss model=%s requested=%s resolved=%s backend=%s device=%s compute_type=%s cpu_threads=%s",
            model_name,
            engine.requested,
            engine.resolved,
            engine.backend,
            device,
            compute_type,
            cpu_threads,
        )
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
    else:
        logger.info(
            "whisper_model_init cache_hit model=%s requested=%s resolved=%s backend=%s device=%s compute_type=%s cpu_threads=%s",
            model_name,
            engine.requested,
            engine.resolved,
            engine.backend,
            device,
            compute_type,
            cpu_threads,
        )
    return _MODEL_CACHE[cache_key]


def _canonicalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().rstrip("/")
    if normalized.lower().startswith("https://huggingface.co/"):
        normalized = normalized[len("https://huggingface.co/") :].strip("/")
    return normalized


def _normalize_model_name(model_name: str) -> str:
    normalized = _canonicalize_model_name(model_name)
    alias = _MODEL_NAME_ALIASES.get(normalized.lower())
    if alias:
        logger.info("whisper_model_alias requested=%s resolved=%s", model_name, alias)
        return alias
    return normalized


def _is_unsupported_gpu_compute_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "do not support efficient int8_float16 computation" in message
        or "no cuda-capable device is detected" in message
        or "failed to create cublas handle" in message
        or "cuda" in message
    )


def _gpu_compute_type_candidates() -> List[str]:
    return ["int8_float16", "float16", "int8"]


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
    require_gpu: bool = False,
    source_filename: Optional[str] = None,
) -> TranscriptionResult:
    requested = _resolve_device()
    engine = _resolve_engine(requested)
    model_name = _normalize_model_name(model_name)
    logger.info(
        "whisper_engine_resolved requested=%s resolved=%s backend=%s reason=%s require_gpu=%s model=%s",
        engine.requested,
        engine.resolved,
        engine.backend,
        engine.reason,
        require_gpu,
        model_name,
    )

    if require_gpu and requested != "intel_gpu":
        raise GpuNotAvailableError("require_gpu_needs_WHISPER_DEVICE=intel_gpu")

    temp_suffix = Path(source_filename or "").suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
        if engine.resolved == "intel_gpu":
            model = None
            init_errors: List[str] = []
            for gpu_compute_type in _gpu_compute_type_candidates():
                try:
                    logger.info("whisper_gpu_init_try model=%s compute_type=%s", model_name, gpu_compute_type)
                    model = _get_model(model_name, engine, compute_type=gpu_compute_type)
                    logger.info("whisper_gpu_init_ok model=%s compute_type=%s", model_name, gpu_compute_type)
                    break
                except Exception as exc:
                    init_errors.append(f"{gpu_compute_type}:{exc.__class__.__name__}:{exc}")
                    logger.warning(
                        "whisper_gpu_init_failed model=%s compute_type=%s err=%s",
                        model_name,
                        gpu_compute_type,
                        exc,
                    )
                    if not _is_unsupported_gpu_compute_error(exc):
                        raise

            if model is None:
                reason = f"{engine.reason};model_init_failed:{' | '.join(init_errors)}"
                if require_gpu:
                    raise GpuNotAvailableError(reason)
                logger.warning("whisper_gpu_fallback reason=%s", reason)
                engine = _fallback_to_cpu(engine, reason)
                model = _get_model(model_name, engine)
        else:
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
            if require_gpu:
                raise GpuNotAvailableError(f"{engine.reason};transcribe_failed:{exc.__class__.__name__}:{exc}") from exc
            logger.warning(
                "whisper_gpu_transcribe_failed fallback_to_cpu err_type=%s err=%s",
                exc.__class__.__name__,
                exc,
            )
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
                SegmentResult(id=idx, start=float(seg.start), end=float(seg.end), text=cleaned)
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
