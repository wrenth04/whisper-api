# Whisper API (OpenAI-Compatible)

此專案提供與 OpenAI/Groq 音訊轉錄格式相容的 FastAPI 服務，端點：

- `POST /v1/audio/transcriptions`

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

目前非 macOS 平台統一使用 **whisper + ctranslate2 (faster-whisper)**，已移除 OpenVINO 依賴。

## 執行

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

或使用內建 CLI：

```bash
python -m app.main --host 0.0.0.0 --port 8000
```

## 裝置選擇與回退策略

### Linux / Windows (`app/transcribe.py`)

- `WHISPER_DEVICE=auto`（預設）：直接使用 CPU（`ctranslate2`）。
- `WHISPER_DEVICE=intel_gpu`：嘗試以 `ctranslate2` GPU 路徑初始化（`device=auto` + 多種 `compute_type`）。失敗時可回退 CPU。
- `WHISPER_DEVICE=cpu`：強制 CPU。

> `intel_gpu` 僅保留相容參數名稱；底層已不再使用 OpenVINO。

### macOS ARM64 (`app/transcribe_mac.py`)

- 優先走 MLX (`mlx-whisper`)。
- 失敗時回退 CPU (`ctranslate2`)。

## Debug 與 GPU 驗證

- `include_debug=true`：回應附上 `requested` / `resolved` / `backend` / `reason`。
- `require_gpu=true`：若未成功使用 GPU 會回 `400` 與 `gpu_not_available`。

`--check-gpu` 在非 macOS 路徑目前回傳：

```json
{"gpu_supported": false, "reason": "gpu_probe_not_supported_without_openvino"}
```

（因為已移除 OpenVINO 預探測邏輯。）

## 錯誤格式

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "code": "..."
  }
}
```
