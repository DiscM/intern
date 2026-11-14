@echo off
rem Bootstraps .venv on the first run, reinstalls only when requirements.txt
rem changed (a copy is kept in .venv), then runs run.py. No cd: paths stay
rem relative to the user's folder.

if not exist .venv\Scripts\python.exe (
  echo creating .venv
  py -3 -m venv .venv 2>nul || python -m venv .venv
)

if not exist .venv\.requirements.txt (
  echo installing requirements
  .venv\Scripts\python -m pip install --quiet -r requirements.txt
  copy /y requirements.txt .venv\.requirements.txt >nul
) else (
  fc /b requirements.txt .venv\.requirements.txt >nul 2>nul
  if errorlevel 1 (
    echo installing requirements
    .venv\Scripts\python -m pip install --quiet -r requirements.txt
    copy /y requirements.txt .venv\.requirements.txt >nul
  )
)

.venv\Scripts\python run.py %*
