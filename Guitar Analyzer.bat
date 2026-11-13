@echo off
rem Starts the Guitar Analyzer launcher; sets up the Python environment on first run.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: creating the Python environment. This takes a minute...
    python -m venv .venv || goto :nopython
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :failed
)
start "" ".venv\Scripts\pythonw.exe" launcher.py
exit /b 0

:nopython
echo Python 3.10 or newer is required: https://www.python.org/downloads/
pause
exit /b 1

:failed
echo Installing the required packages failed. Check your internet connection and try again.
pause
exit /b 1
