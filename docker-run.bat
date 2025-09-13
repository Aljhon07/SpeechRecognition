@echo off
REM Speech Recognition Docker Management Script for Windows
set IMAGE_NAME=sephirah07/lingua-speech:dev

if "%1"=="build" (
    echo Building Speech Recognition Docker image...
    docker build -t %IMAGE_NAME% .
    echo Build completed!
    goto :eof
)

if "%1"=="start" (
    echo Starting Speech Recognition application...
    docker-compose up -d %IMAGE_NAME% 
    echo Application started! Access it at http://localhost:8080
    goto :eof
)

if "%1"=="stop" (
    echo Stopping Speech Recognition application...
    docker-compose down
    goto :eof
)

if "%1"=="restart" (
    echo Restarting Speech Recognition application...
    docker-compose restart %IMAGE_NAME% 
    goto :eof
)

if "%1"=="logs" (
    echo Showing application logs...
    docker-compose logs -f %IMAGE_NAME% 
    goto :eof
)

if "%1"=="train" (
    echo Running training with TTY and service volumes...
    docker run --rm --tty ^
        -v "%cd%\output:/app/output" ^
        -v "%cd%\logs:/app/logs" ^
        -v "%cd%\commonvoice:/app/commonvoice" ^
        -v "%cd%\src:/app/src" ^
        -v "%cd%\inference:/app/inference" ^
        -v "%cd%\tools:/app/tools" ^
        -v "%cd%\config.py:/app/config.py" ^
        -v "%cd%\main.py:/app/main.py" ^
        -e PYTHONPATH=/app ^
        -e GENAI_API_KEY=%GENAI_API_KEY% ^
        %IMAGE_NAME%  python main.py
    goto :eof
)

if "%1"=="preprocess" (
    echo Running preprocessing with TTY and service volumes...
    docker run --rm --tty ^
        -v "%cd%\output:/app/output" ^
        -v "%cd%\logs:/app/logs" ^
        -v "%cd%\commonvoice:/app/commonvoice" ^
        -v "%cd%\commonvoice:/app/librispeech" ^
        -v "%cd%\src:/app/src" ^
        -v "%cd%\inference:/app/inference" ^
        -v "%cd%\tools:/app/tools" ^
        -v "%cd%\config.py:/app/config.py" ^
        -e PYTHONPATH=/app ^
        -e GENAI_API_KEY=%GENAI_API_KEY% ^
        %IMAGE_NAME%  python -c "from src.preprocess import preprocess; preprocess()"
    goto :eof
)

if "%1"=="extract" (
    echo Running Whisper feature extraction...
    docker run --rm --tty ^
        -v "%cd%\output:/app/output" ^
        -v "%cd%\logs:/app/logs" ^
        -v "%cd%\commonvoice:/app/commonvoice" ^
        -v "%cd%\src:/app/src" ^
        -v "%cd%\inference:/app/inference" ^
        -v "%cd%\tools:/app/tools" ^
        -v "%cd%\config.py:/app/config.py" ^
        -v "%cd%\librispeech:/app/librispeech" ^
        -e PYTHONPATH=/app ^
        -e GENAI_API_KEY=%GENAI_API_KEY% ^
        %IMAGE_NAME%  python tools/whisper_extractor.py
    goto :eof
)

if "%1"=="shell" (
    echo Opening shell in container...
    docker-compose exec %IMAGE_NAME%  bash
    goto :eof
)

if "%1"=="clean" (
    echo Cleaning up Docker containers and images...
    docker-compose down --rmi all --volumes --remove-orphans
    goto :eof
)

echo Speech Recognition Docker Management
echo Usage: %0 {build^|start^|stop^|restart^|logs^|train^|preprocess^|shell^|clean}
echo.
echo Commands:
echo   build      - Build the Docker image
echo   start      - Start the application
echo   stop       - Stop the application
echo   restart    - Restart the application
echo   logs       - Show application logs
echo   train      - Run training with TTY (real-time progress)
echo   preprocess - Run preprocessing with TTY (real-time progress)
echo   shell      - Open bash shell in container
echo   clean      - Clean up all Docker resources
echo.
echo Configuration:
echo   Edit .env file to switch between development/production mode
echo   Comment/uncomment source volume mounts in docker-compose.yml for production
