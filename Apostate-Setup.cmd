@echo off
REM double-click installer: runs the setup wizard
SET COLORTERM=truecolor
SET TERM=xterm-256color
node "%~dp0setup.js"
echo.
pause
