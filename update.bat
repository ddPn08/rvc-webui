@echo off
if exist .git goto :reset

git init
git remote add origin https://github.com/ddPn08/rvc-webui.git

:reset
git fetch --prune
git reset --hard origin/main
pause