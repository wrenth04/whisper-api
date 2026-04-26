# Whisper API (OpenAI-Compatible)

此專案提供一個與 OpenAI/Groq 音訊轉錄格式接近的 FastAPI 服務，端點為：

- `POST /v1/audio/transcriptions`

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 打包/部署前必要 include 檢查

此專案的必要 runtime 相依已包含在 `requirements.txt`（`faster-whisper`、`openvino`、`python-multipart` 等）。建議在打包前先做一次 import smoke test：

```bash
python -c "import fastapi, faster_whisper, openvino; print('imports_ok')"
```

## 執行

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

若你是打包成 exe，現在可直接用：

```bash
./whisper-api --check-gpu
```

此模式只會做 GPU 探測並立即結束，不會啟動 API 服務。回傳範例：

```json
{"gpu_supported": false, "reason": "openvino_gpu_missing:['CPU']"}
```

## Windows 打包（PyInstaller）

如果你在 Windows 上遇到「Python 可以啟動，但 exe 的 `--check-gpu` 回傳 `openvino_gpu_missing:[]`」，通常是打包時少帶了 OpenVINO/faster-whisper 的 runtime 檔。

本專案提供一個可直接執行的腳本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_exe.ps1
```

完成後執行：

```powershell
.\dist\whisper-api-server.exe --check-gpu
```

建議判斷：
- 若輸出包含 `openvino_gpu_available`：代表 exe 可見 GPU。
- 若仍是 `openvino_gpu_missing:[]`：優先檢查 Intel 顯示驅動與執行環境是否真的可見 GPU（例如遠端桌面/虛擬機可能隱藏裝置）。

也可先在同機器 Python 環境驗證：

```powershell
python -c "from openvino import Core; print(Core().available_devices)"
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

## GPU 支援確認參數（新增）

API 新增 `require_gpu`（`true/false`，預設 `false`）：

- `require_gpu=false`：維持原本策略，可回退 CPU。
- `require_gpu=true`：若無法解析為 Intel GPU，會直接回 `400 gpu_not_available`，可用於 CI 或部署驗收。

建議驗證指令：

```bash
WHISPER_DEVICE=intel_gpu curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -X POST \
  -F "file=@/path/to/audio.wav" \
  -F "model=small" \
  -F "response_format=json" \
  -F "include_debug=true" \
  -F "require_gpu=true"
```

判斷方式：
- 成功：`debug.resolved` 為 `intel_gpu`（且 `backend` 為 `openvino`）。
- 失敗：回傳 `400`，`error.code = "gpu_not_available"`，訊息包含原因（例如 `openvino_gpu_missing`）。

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
