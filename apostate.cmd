@echo off
REM color
SET COLORTERM=truecolor
SET TERM=xterm-256color
REM no args -> tui; ablate/test/talk/list -> command
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
python -m apostate %*
