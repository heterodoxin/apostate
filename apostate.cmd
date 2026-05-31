@echo off
setlocal enabledelayedexpansion

REM color
SET COLORTERM=truecolor
SET TERM=xterm-256color

REM resize
mode con: cols=240 lines=60 2>nul
powershell -NoProfile -Command "[Console]::BufferWidth=240; [Console]::WindowWidth=240; [Console]::BufferHeight=60; [Console]::WindowHeight=60" 2>nul

REM launch
node "%~dp0main.js" %*
