@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
set "LABELME_PATH=%PROJECT_DIR%labelme"

where pixi >nul 2>nul
if errorlevel 1 (
    echo Pixi was not found in PATH.
    echo Install Pixi first, then run this script again.
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%"
if errorlevel 1 (
    echo Cannot open the Labelme project directory.
    pause
    exit /b 1
)

echo Checking the Pixi environment...
pixi install
if errorlevel 1 goto :failed

set "OSAM_PATH="
for /f "usebackq delims=" %%I in (`pixi run python -c "import os, osam; print(os.path.dirname(osam.__file__))"`) do set "OSAM_PATH=%%I"
if not defined OSAM_PATH (
    echo Could not locate the osam package in the Pixi environment.
    goto :failed
)

echo Building Labelme with PyInstaller...
set "PYDANTIC_DISABLE_PLUGINS=__all__"
pixi run pyinstaller packaging_entry.py ^
    --name=Labelme ^
    --windowed ^
    --noconfirm ^
    --paths "%PROJECT_DIR%." ^
    --collect-submodules labelme ^
    --specpath=build ^
    --add-data "%OSAM_PATH%\_models\yoloworld\clip\bpe_simple_vocab_16e6.txt.gz;osam\_models\yoloworld\clip" ^
    --add-data "%LABELME_PATH%\_config\default_config.yaml;labelme\_config" ^
    --add-data "%LABELME_PATH%\icons\*;labelme\icons" ^
    --add-data "%LABELME_PATH%\translate\*;translate" ^
    --icon "%LABELME_PATH%\icons\icon-256.png" ^
    --onedir
if errorlevel 1 goto :failed

echo.
echo Build completed successfully.
echo Output: %PROJECT_DIR%dist\Labelme
popd
exit /b 0

:failed
set "LABELME_EXIT_CODE=%ERRORLEVEL%"
if "%LABELME_EXIT_CODE%"=="0" set "LABELME_EXIT_CODE=1"
popd
echo.
echo Build failed with error code %LABELME_EXIT_CODE%.
pause
exit /b %LABELME_EXIT_CODE%
