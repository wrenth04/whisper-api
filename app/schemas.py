from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class TranscriptionDebug(BaseModel):
    requested: str
    resolved: str
    backend: str
    reason: str


class JsonTranscriptionResponse(BaseModel):
    text: str
    debug: Optional[TranscriptionDebug] = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class VerboseJsonTranscriptionResponse(BaseModel):
    task: Literal["transcribe"] = "transcribe"
    language: Optional[str] = None
    duration: Optional[float] = None
    text: str
    segments: List[Segment] = Field(default_factory=list)
    debug: Optional[TranscriptionDebug] = None
