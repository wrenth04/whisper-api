from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from faster_whisper import WhisperModel


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


_MODEL_CACHE: dict[str, WhisperModel] = {}


def _get_model(model_name: str) -> WhisperModel:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = WhisperModel(model_name)
    return _MODEL_CACHE[model_name]


def transcribe_audio(
    audio_bytes: bytes,
    model_name: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> TranscriptionResult:
    model = _get_model(model_name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
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

        return TranscriptionResult(
            text=" ".join(texts).strip(),
            language=getattr(info, "language", language),
            duration=getattr(info, "duration", None),
            segments=segment_results,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
