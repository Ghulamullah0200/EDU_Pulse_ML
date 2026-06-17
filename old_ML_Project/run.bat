@echo off
title Student Academic Risk Project
echo Installing packages and starting the app...
python -m pip install pandas scikit-learn matplotlib streamlit joblib numpy --quiet
python -m streamlit run app.py
pause
