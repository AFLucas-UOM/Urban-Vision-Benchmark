@echo off
setlocal
cd /d "%~dp0\..\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "Scripts\LabelStudio\start_labelstudio.ps1" %*
