@echo off
setlocal

REM === detect venv activation script ===
if exist ".venv\Scripts\activate.bat" (
  set "VENV_ACT=.venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
  set "VENV_ACT=venv\Scripts\activate.bat"
) else (
  echo Virtual environment not found. Create one or edit this script to point to your venv.
  pause
  exit /b 1
)

REM === Worker runtime env (tune these as needed) ===
set "WORKER_DEVICE=cuda"
set "WHISPER_MODEL=tiny"
set "WHISPER_COMPUTE_TYPE=float16"

REM Alignment / NER / debug tuning (change values as needed)
set "DEBUG_NER=1"
set "ALIGN_OVERLAP_RATIO_MIN=0.7"
set "ALIGN_OVERLAP_MIN=0.02"
set "ALIGN_MAX_GAP=0.5"
set "NEAREST_MAX_DISTANCE=0.45"
set "UNASSIGNED_MERGE_MAX_GAP=0.25"
set "TRUECASE=1"
set "MAX_UTTERANCES_TO_SCAN_FOR_NAMES=300"

REM === Start worker server in a new window ===
start "" cmd /k "%VENV_ACT% && set WORKER_DEVICE=%WORKER_DEVICE% && set WHISPER_MODEL=%WHISPER_MODEL% && set WHISPER_COMPUTE_TYPE=%WHISPER_COMPUTE_TYPE% && set DEBUG_NER=%DEBUG_NER% && set ALIGN_OVERLAP_RATIO_MIN=%ALIGN_OVERLAP_RATIO_MIN% && set ALIGN_OVERLAP_MIN=%ALIGN_OVERLAP_MIN% && set ALIGN_MAX_GAP=%ALIGN_MAX_GAP% && set NEAREST_MAX_DISTANCE=%NEAREST_MAX_DISTANCE% && set UNASSIGNED_MERGE_MAX_GAP=%UNASSIGNED_MERGE_MAX_GAP% && set TRUECASE=%TRUECASE% && set MAX_UTTERANCES_TO_SCAN_FOR_NAMES=%MAX_UTTERANCES_TO_SCAN_FOR_NAMES% && python -m uvicorn worker_server:app --host 0.0.0.0 --port 9000 --workers 1"

REM === Start Flask app in a new window ===
start "" cmd /k "%VENV_ACT% && python app.py"

REM Give Flask a moment to boot, then open browser (optional)
timeout /t 5 >nul
start "" http://127.0.0.1:7860

endlocal
