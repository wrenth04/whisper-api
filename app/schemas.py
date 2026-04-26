from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class JsonTranscriptionResponse(BaseModel):
    text: str


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
