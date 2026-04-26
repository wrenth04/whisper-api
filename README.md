# whisper api

## 從 Actions 下載 Windows EXE

1. 前往 GitHub 專案頁面的 **Actions**。
2. 開啟 `Build Windows EXE` workflow 的成功執行紀錄。
3. 在頁面下方 **Artifacts** 區塊下載 `whisper-api-server-windows-exe`。
4. 解壓後可取得 `whisper-api-server.exe`。

> 此 workflow 會在 `main` branch 的 push 以及 `v*` tag push 時觸發；tag 觸發時也會把 `dist/whisper-api-server.exe` 附加到 GitHub Release。
