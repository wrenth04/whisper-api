# whisper api

## 從 Actions 下載 Windows EXE

1. 前往 GitHub 專案頁面的 **Actions**。
2. 開啟 `Build Windows EXE` workflow 的成功執行紀錄。
3. 在頁面下方 **Artifacts** 區塊下載 `whisper-api-server-windows-exe`。
4. 解壓後可取得 `whisper-api-server.exe`。

> 此 workflow 會在 `main` branch 的 push 以及 `v*` tag push 時觸發；tag 觸發時也會把 `dist/whisper-api-server.exe` 附加到 GitHub Release。
# Whisper API

此專案提供 Whisper 語音轉文字服務，並在啟動時優先嘗試 **Intel GPU (OpenVINO Runtime)**，不可用時自動 fallback 到 CPU。

## 裝置策略（`app/transcribe.py`）

- `WHISPER_DEVICE=auto`（預設）
  - 先探測 OpenVINO 是否可用且可見 `GPU` 裝置。
  - 可用則使用 Intel GPU；否則自動降級到 CPU。
- `WHISPER_DEVICE=intel_gpu`
  - 仍會先走 OpenVINO GPU 探測；若失敗目前採穩定優先，降級到 CPU。
- `WHISPER_DEVICE=cpu`
  - 直接使用 CPU，不探測 GPU。

## 環境變數

- `WHISPER_DEVICE=auto|intel_gpu|cpu`
- `WHISPER_MODEL_SIZE=tiny|base|small|medium|large-v3|...`

範例：

```bash
export WHISPER_DEVICE=auto
export WHISPER_MODEL_SIZE=small
```

## 啟動時日誌與 API debug 欄位

`WhisperEngine` 建立時會輸出實際使用裝置與後端，例如：

- requested device
- resolved device
- backend
- model size
- fallback reason

在 API 層呼叫 `transcribe(..., include_debug=True)` 時，回應可包含：

```json
{
  "text": "...",
  "model_size": "small",
  "debug": {
    "device": "intel_gpu",
    "backend": "openvino",
    "requested_device": "auto",
    "reason": "openvino_gpu_available"
  }
}
```

## Intel 顯卡（Windows）先決條件

若要穩定使用 Intel GPU，建議確認：

1. **Intel 顯示驅動已更新**（Arc / Iris Xe / UHD 對應版本）。
2. **OpenVINO Runtime 已安裝**（Python 套件 `openvino`，版本與 Python 相容）。
3. 系統可被 OpenVINO 探測到 GPU（`Core().available_devices` 含 `GPU`）。
4. 建議使用 64-bit Python 與對應 Visual C++ Runtime。

## 常見錯誤排查

1. `openvino.runtime_not_installed`
   - 安裝 OpenVINO：`pip install openvino`

2. `openvino_probe_failed: ...`
   - 通常是驅動或 Runtime 相容性問題。
   - 先更新 Intel 顯示驅動，再重新安裝 OpenVINO。

3. `openvino_devices=['CPU']`
   - OpenVINO 可用，但看不到 GPU。
   - 檢查 BIOS / 驅動 / 遠端桌面環境是否影響 GPU 可見性。

4. 明確設定 `WHISPER_DEVICE=intel_gpu` 但仍落到 CPU
   - 目前策略為「穩定優先」：GPU 不可用即 fallback CPU，避免服務啟動失敗。

## 推理後端方案

本專案採用 **方案 A（建議）**：Whisper + OpenVINO Runtime。理由：

- 與 Intel GPU 相容性成熟。
- 當 GPU 不可用時可平滑退回 CPU。
- 部署與排錯路徑明確（驅動 + OpenVINO）。
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
