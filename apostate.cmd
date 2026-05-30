@echo off
setlocal enabledelayedexpansion

REM Set true color support
SET COLORTERM=truecolor
SET TERM=xterm-256color

REM Resize console window (try multiple methods)
mode con: cols=240 lines=60 2>nul
powershell -NoProfile -Command "[Console]::BufferWidth=240; [Console]::WindowWidth=240; [Console]::BufferHeight=60; [Console]::WindowHeight=60" 2>nul

REM Launch Node TUI
node "%~dp0tui.js" %*
