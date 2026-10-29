@echo off
cd /d "%~dp0"
set "AIRTRUST_PY=python"
if exist .venv\Scripts\python.exe set "AIRTRUST_PY=.venv\Scripts\python.exe"
%AIRTRUST_PY% scripts\run_experiments.py
if errorlevel 1 goto failed
%AIRTRUST_PY% scripts\run_spatial_cv.py
if errorlevel 1 goto failed
%AIRTRUST_PY% scripts\run_calibration.py
if errorlevel 1 goto failed
%AIRTRUST_PY% scripts\run_sensitivity.py
if errorlevel 1 goto failed
%AIRTRUST_PY% -m pytest
if errorlevel 1 goto failed
%AIRTRUST_PY% scripts\build_extensions.py
if errorlevel 1 goto failed
%AIRTRUST_PY% scripts\build_submission.py
if errorlevel 1 goto failed
%AIRTRUST_PY% scripts\package_submission.py
if errorlevel 1 goto failed
%AIRTRUST_PY% scripts\verify_submission.py
if errorlevel 1 goto failed
echo Reproduction and packaging completed.
pause
exit /b 0
:failed
echo A step failed. Review the error above before submitting.
pause
exit /b 1
