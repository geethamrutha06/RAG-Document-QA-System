@echo off
echo =====================================
echo RAG Document Q&A System Setup
echo =====================================

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

if not exist .env (
    copy .env.example .env
    echo.
    echo WARNING: Please edit .env file and add your OpenAI API key
    echo You can get one from: https://platform.openai.com/api-keys
    echo.
)

echo.
echo =====================================
echo Setup complete!
echo =====================================
echo.
echo To run the application:
echo   venv\Scripts\activate
echo   streamlit run app.py
echo.
pause
