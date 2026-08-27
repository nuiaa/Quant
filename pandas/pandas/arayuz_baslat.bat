@echo off
echo ==========================================
echo Quant Master - Command Center Baslatiliyor
echo ==========================================
echo Lutfen bekleyin... Arka plan hazirliklari yapiliyor.

"C:\Users\cromagnon\AppData\Local\Python\pythoncore-3.14-64\python.exe" arayuz.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [HATA] Bir sorun olustu! Arayuz baslatilamadi.
    pause
)
