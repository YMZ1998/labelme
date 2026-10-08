```bat
@echo off
setlocal

set "LABELME_CONFIG_INI=%~dp0config.ini"

pushd "%~dp0"
if errorlevel 1 (
    echo Cannot open the Labelme project directory.
    pause
    exit /b 1
)

rem Avoid native Qt/driver crashes during canvas painting on some Windows GPUs.
if not defined QT_OPENGL set "QT_OPENGL=software"
if not defined QT_QUICK_BACKEND set "QT_QUICK_BACKEND=software"

pixi run python -m labelme %*
set "LABELME_EXIT_CODE=%ERRORLEVEL%"

popd

if not "%LABELME_EXIT_CODE%"=="0" (
    echo.
    echo Labelme exited with error code %LABELME_EXIT_CODE%.
    pause
)

exit /b %LABELME_EXIT_CODE%
```
