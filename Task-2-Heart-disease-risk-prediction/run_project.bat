@echo off
REM Run this script instead of clicking "Run All" manually.
REM It runs the full notebook AND updates the HTML export together, every time.
REM
REM Usage (from the project root folder, double-click this file or run in Command Prompt):
REM   run_project.bat

cd notebooks

echo Running the notebook...
python -m nbconvert --to notebook --execute --inplace heart_disease_risk_prediction.ipynb

echo Updating the HTML export...
python -m nbconvert --to html heart_disease_risk_prediction.ipynb

echo.
echo Done! Both files are now up to date:
echo   - notebooks\heart_disease_risk_prediction.ipynb
echo   - notebooks\heart_disease_risk_prediction.html
pause
