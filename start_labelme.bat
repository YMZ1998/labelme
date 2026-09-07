@echo off
setlocal
set "LABELME_ENV=D:\conda\envs\labelme"
set "LABELME_PYTHON=%LABELME_ENV%\python.exe"

if not exist "%LABELME_PYTHON%" (
    echo Python was not found: "%LABELME_PYTHON%"
    pause
    exit /b 1
)

pushd "%~dp0"
if errorlevel 1 (
    echo Cannot open the Labelme project directory.
    pause
    exit /b 1
)

set "PATH=%LABELME_ENV%;%LABELME_ENV%\Library\bin;%LABELME_ENV%\Scripts;%PATH%"
"%LABELME_PYTHON%" -m labelme %*
set "LABELME_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%LABELME_EXIT_CODE%"=="0" (
    echo.
    echo Labelme exited with error code %LABELME_EXIT_CODE%.
    pause
)
exit /b %LABELME_EXIT_CODE%
