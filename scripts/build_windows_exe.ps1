$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

pyinstaller `
  --noconfirm `
  --clean `
  --onefile `
  --name whisper-api-server `
  --collect-all faster_whisper `
  --hidden-import faster_whisper `
  app/main.py

Write-Host ""
Write-Host "Build done. Binary: dist/whisper-api-server.exe"
Write-Host "GPU check: .\\dist\\whisper-api-server.exe --check-gpu"
