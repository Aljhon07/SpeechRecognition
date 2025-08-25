@echo off
echo Building Docker image for Speech Recognition...
docker build -t speech-recognition .

echo.
echo Running preprocessing with TTY for real-time progress...
docker run --tty -v "%cd%:/app" speech-recognition python -c "from src.preprocess import preprocess; preprocess()"

echo.
echo Preprocessing completed!
pause
