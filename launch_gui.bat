@echo off
setlocal
cd /d "%~dp0"

:: Request Administrator privileges (required for Scapy live packet capture)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting Administrator privileges for Live Packet Capture...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb RunAs"
    exit /b
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m flowpic.gui
) else (
  echo Virtual environment not found. Please create .venv and install dependencies first.
  pause
)
