@echo off
setlocal
cd /d "%~dp0\..\..\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "Scripts\MTSD-Scripts\LabelStudio\start_labelstudio.ps1" %*
