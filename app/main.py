from __future__ import annotations

import argparse
import json
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
from app.transcribe import GpuNotAvailableError, check_gpu_support, transcribe_audio

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
    include_debug: bool = Form(False),
    require_gpu: bool = Form(False),
):
    if not file.filename:
        raise ApiError("No audio file was provided.", code="missing_file")

    if temperature < 0 or temperature > 1:
        raise ApiError("temperature must be between 0 and 1.", code="invalid_temperature")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise ApiError("Uploaded file is empty.", code="empty_file")

    try:
        result = transcribe_audio(
            audio_bytes=audio_bytes,
            model_name=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            include_debug=include_debug,
            require_gpu=require_gpu,
        )
    except GpuNotAvailableError as exc:
        raise ApiError(
            f"GPU check failed: {exc.reason}",
            code="gpu_not_available",
            status_code=400,
        ) from exc

    if response_format == "json":
        return JsonTranscriptionResponse(text=result.text, debug=result.debug)

    return VerboseJsonTranscriptionResponse(
        language=result.language,
        duration=result.duration,
        text=result.text,
        segments=[
            Segment(id=s.id, start=s.start, end=s.end, text=s.text) for s in result.segments
        ],
        debug=result.debug,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper OpenAI-Compatible API")
    parser.add_argument("--check-gpu", action="store_true", help="Only check GPU support and exit.")
    parser.add_argument("--host", default="0.0.0.0", help="Host for uvicorn server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for uvicorn server.")
    args = parser.parse_args()

    if args.check_gpu:
        supported, reason = check_gpu_support()
        payload = {"gpu_supported": supported, "reason": reason}
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if supported else 2

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
