@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo PRISM environment missing. Run: uv sync --locked
    exit /b 1
)
".venv\Scripts\python.exe" -m prism_app %*
exit /b %errorlevel%
