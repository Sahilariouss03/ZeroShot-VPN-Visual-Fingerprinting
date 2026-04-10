@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m flowpic.gui
) else (
  echo Virtual environment not found. Please create .venv and install dependencies first.
  pause
)
