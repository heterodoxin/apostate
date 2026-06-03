@echo off
REM double-click installer
SET COLORTERM=truecolor
SET TERM=xterm-256color
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
python -m apostate setup
echo.
pause
