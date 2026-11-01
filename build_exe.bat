@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   Building FontStudio.exe
echo ============================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run this file again.
  pause & exit /b 1
)
python -c "import tkinter" 2>nul
if errorlevel 1 (
  echo This Python has no Tk. Re-run the python.org installer, choose Modify, and tick "tcl/tk and IDLE".
  pause & exit /b 1
)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 ( echo Package install failed. & pause & exit /b 1 )
python tests\test_engine.py
if errorlevel 1 ( echo Self-test failed - see messages above. & pause & exit /b 1 )
python -m PyInstaller --noconfirm --clean --onefile --windowed --name FontStudio ^
  --collect-submodules fontTools --hidden-import brotli fontstudio.py
if errorlevel 1 ( echo Build failed. & pause & exit /b 1 )
echo.
echo Done!  Your program is:  %cd%\dist\FontStudio.exe
pause
