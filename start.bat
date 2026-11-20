@echo off
REM Double-click this (or run `start.bat` from a terminal) to bring up
REM MediaMTX + backend + frontend together. See run_all.py for flags,
REM e.g. `start.bat --skip-mediamtx` if you already have MediaMTX running.
cd /d "%~dp0"
python run_all.py %*
pause
