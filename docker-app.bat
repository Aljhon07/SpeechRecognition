@echo off
echo Building Docker image for Speech Recognition...
docker build -t speech-recognition .

echo.
echo Starting Flask application with TTY...
docker run --tty -p 5000:8080 -v "%cd%:/app" speech-recognition python -m inference.app

echo.
echo Application stopped!
pause
