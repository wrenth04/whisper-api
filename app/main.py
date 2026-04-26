from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.schemas import (
    ErrorBody,
    ErrorResponse,
    JsonTranscriptionResponse,
    Segment,
    VerboseJsonTranscriptionResponse,
)
from app.transcribe import transcribe_audio

app = FastAPI(title="Whisper OpenAI-Compatible API")


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: Optional[str] = None,
        status_code: int = 400,
    ) -> None:
        self.message = message
        self.error_type = error_type
        self.code = code
        self.status_code = status_code
        super().__init__(message)


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(message=exc.message, type=exc.error_type, code=exc.code))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            message=str(exc.detail),
            type="invalid_request_error",
            code=str(exc.status_code),
        )
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            message=f"Internal server error: {exc}",
            type="server_error",
            code="internal_error",
        )
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Literal["json", "verbose_json"] = Form("json"),
    temperature: float = Form(0.0),
):
    if not file.filename:
        raise ApiError("No audio file was provided.", code="missing_file")

    if temperature < 0 or temperature > 1:
        raise ApiError("temperature must be between 0 and 1.", code="invalid_temperature")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise ApiError("Uploaded file is empty.", code="empty_file")

    result = transcribe_audio(
        audio_bytes=audio_bytes,
        model_name=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
    )

    if response_format == "json":
        return JsonTranscriptionResponse(text=result.text)

    return VerboseJsonTranscriptionResponse(
        language=result.language,
        duration=result.duration,
        text=result.text,
        segments=[
            Segment(id=s.id, start=s.start, end=s.end, text=s.text) for s in result.segments
        ],
    )
