@echo off
echo Starting FX Bot...
start "FX Bot (Watchdog)" cmd /k "cd /d %~dp0 && py watchdog.py"

timeout /t 3 /nobreak >nul

echo Starting Dashboard...
start "FX Dashboard" cmd /k "cd /d %~dp0 && py dashboard.py"

echo.
echo Both processes started in separate windows.
echo Close those windows to stop them.
