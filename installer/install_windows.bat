@echo off
REM ============================================================
REM  Market Analysis & Trade Signal Software - Windows Installer
REM  Double-click this file. It creates a desktop shortcut.
REM ============================================================
setlocal
cd /d "%~dp0.."

echo.
echo ============================================
echo   Installing Trading Software (Windows)
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python nahi mila!
    echo.
    echo 1. https://www.python.org/downloads/ se Python 3.10+ install karo
    echo 2. INSTALLER ME "Add python.exe to PATH" KA TICK ZAROOR LAGAO
    echo 3. Phir ye file dubara double-click karo
    echo.
    pause
    exit /b 1
)

echo [1/4] Python virtual environment bana raha hoon...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] venv create nahi hua. "python -m pip install --upgrade pip" try karo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [2/4] Packages install ho rahe hain (2-5 min)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install fail hua. Internet check karo aur dubara chalao.
    pause
    exit /b 1
)

echo [3/4] Demo data taiyar kar raha hoon...
python main.py fetch --symbols SYNTH --timeframes 1h --provider synthetic --bars 9000 >nul 2>nul

echo [4/4] Desktop shortcut bana raha hoon...
set "VBS=%TEMP%\trading_shortcut.vbs"
> "%VBS%" echo Set oWS = WScript.CreateObject("WScript.Shell")
>> "%VBS%" echo Set oLink = oWS.CreateShortcut(oWS.SpecialFolders("Desktop") ^& "\Trading Software.lnk")
>> "%VBS%" echo oLink.TargetPath = "%CD%\.venv\Scripts\pythonw.exe"
>> "%VBS%" echo oLink.Arguments = "Launch.py"
>> "%VBS%" echo oLink.WorkingDirectory = "%CD%"
>> "%VBS%" echo oLink.Description = "Market Analysis and Trade Signal Software"
>> "%VBS%" echo oLink.Save
cscript //nologo "%VBS%" >nul 2>nul
del "%VBS%" >nul 2>nul

echo.
echo ============================================
echo   INSTALL HO GAYA! 
echo ============================================
echo.
echo   Desktop par "Trading Software" shortcut hai - double-click karo.
echo   Browser me dashboard khul jayega.
echo.
echo   REAL MARKET DATA ke liye (first time):
echo     cd %CD%
echo     .venv\Scripts\activate
echo     python main.py fetch --symbols AAPL,MSFT,BTC-USD --timeframes 15m,1h,1d --provider yfinance
echo.
echo   AI ke liye: openrouter.ai/keys se key banao,
echo   app ke sidebar me paste karke SAVE karo.
echo.
echo   Disclaimer: Not financial advice. Based on historical
echo   statistical edge, not a guarantee.
echo.
pause
