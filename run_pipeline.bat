@echo off
setlocal

:: This batch file automates the processing of a specific folder
set INPUT_FOLDER=input_files
set OUTPUT_FOLDER=output_results

echo ========================================================
echo Keyword Classification Automation Pipeline
echo ========================================================
echo Input folder:  %INPUT_FOLDER%
echo Output folder: %OUTPUT_FOLDER%
echo ========================================================

if not exist "%INPUT_FOLDER%" (
    echo [ERROR] Input folder '%INPUT_FOLDER%' not found.
    echo Creating it now...
    mkdir "%INPUT_FOLDER%"
    echo Please place your files in the '%INPUT_FOLDER%' directory and run this script again.
    pause
    exit /b
)

python classify_and_highlight.py --input "%INPUT_FOLDER%" --output "%OUTPUT_FOLDER%"

echo.
echo ========================================================
echo Done! Check the '%OUTPUT_FOLDER%' directory for results.
echo ========================================================
pause
