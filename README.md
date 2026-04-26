# Whisper API (OpenAI-Compatible)

此專案提供一個與 OpenAI/Groq 音訊轉錄格式接近的 FastAPI 服務，端點為：

- `POST /v1/audio/transcriptions`

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 執行

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 裝置選擇與回退策略

`app/transcribe.py` 會讀取 `WHISPER_DEVICE`：

- `auto`（預設）：偵測 OpenVINO + GPU，若不可用自動退回 CPU。
- `intel_gpu`：強制嘗試 Intel GPU 路徑；失敗時退回 CPU（穩定優先）。
- `cpu`：直接走 CPU。

另外，模型建立時會**顯式**傳入 `device` 與 `compute_type`，不依賴預設值。

## OpenVINO 相容矩陣與環境限制

| 元件 | 版本建議 | 說明 |
|---|---|---|
| Python | 3.10–3.12 (64-bit) | 建議使用 64-bit 環境 |
| faster-whisper | `>=1.1.0,<2.0.0` | Whisper 推理 |
| openvino | `>=2024.3.0,<2026.0.0` | OpenVINO Runtime 偵測與 Intel GPU 可見性判斷 |
| Intel GPU Driver | 最新穩定版 | 需可被 OpenVINO `Core().available_devices` 探測到 `GPU` |
| OS/平台 | Linux / Windows x86_64 | 非 x86_64 或缺驅動時通常會退回 CPU |

> 若 OpenVINO 可 import 但看不到 `GPU`，服務會回報原因並改用 CPU。

## Debug 可觀測性

API 新增可選參數 `include_debug`（`true/false`，預設 `false`）。

- `include_debug=true` 時，回應會附上：
  - `requested`：請求裝置策略
  - `resolved`：實際使用裝置
  - `backend`：實際後端
  - `reason`：探測或回退原因

範例：

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "model=small" \
  -F "response_format=json" \
  -F "include_debug=true"
```

回應片段：

```json
{
  "text": "...",
  "debug": {
    "requested": "auto",
    "resolved": "cpu",
    "backend": "ctranslate2",
    "reason": "openvino_gpu_missing:['CPU']"
  }
}
```

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
