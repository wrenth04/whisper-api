# Whisper API (OpenAI-Compatible)

此專案提供一個與 OpenAI/Groq 音訊轉錄格式接近的 FastAPI 服務，端點為：

- `POST /v1/audio/transcriptions`

## 專案結構

```text
.
├── app/
│   ├── main.py         # FastAPI 入口
│   ├── schemas.py      # OpenAI 相容 request/response schema
│   └── transcribe.py   # Whisper 推理邏輯
└── requirements.txt
```

## 本機啟動

1. 安裝依賴

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 啟動 API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## EXE 打包後啟動（範例）

若你將入口打包為可執行檔（例如 `whisper-api.exe`），啟動可參考：

```bash
./whisper-api.exe --host 0.0.0.0 --port 8000
```

> 實際參數取決於你的打包方式（PyInstaller/Nuitka 等）與 CLI 設計。

## API 呼叫範例

### `response_format=json`

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "model=small" \
  -F "language=zh" \
  -F "prompt=以下是會議逐字稿" \
  -F "response_format=json" \
  -F "temperature=0"
```

成功回應範例：

```json
{"text":"這是一段轉錄內容"}
```

### `response_format=verbose_json`

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "model=small" \
  -F "response_format=verbose_json"
```

成功回應範例（最小可用）：

```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 12.34,
  "text": "這是一段轉錄內容",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.4, "text": "這是"}
  ]
}
```

## 錯誤格式

統一回傳格式：

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "code": "..."
  }
}
```
