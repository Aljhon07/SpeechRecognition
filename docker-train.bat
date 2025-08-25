@echo off
echo Building Docker image for Speech Recognition...
docker build -t speech-recognition .

echo.
echo Running training with TTY for real-time progress...
docker run --tty -v "%cd%:/app" speech-recognition python src/train.py

echo.
echo Training completed!
pause
