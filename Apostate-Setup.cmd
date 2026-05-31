@echo off
REM setup
SET COLORTERM=truecolor
SET TERM=xterm-256color
node "%~dp0setup.js"
echo.
pause
