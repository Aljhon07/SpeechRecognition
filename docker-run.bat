@echo off
REM Speech Recognition Docker Management Script for Windows

if "%1"=="build" (
    echo Building Speech Recognition Docker image...
    docker-compose build
    goto :eof
)

if "%1"=="start" (
    echo Starting Speech Recognition application...
    docker-compose up -d speech-recognition
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
    docker-compose restart speech-recognition
    goto :eof
)

if "%1"=="logs" (
    echo Showing application logs...
    docker-compose logs -f speech-recognition
    goto :eof
)

if "%1"=="train" (
    echo Running training in container...
    docker-compose --profile training up training
    goto :eof
)

if "%1"=="shell" (
    echo Opening shell in container...
    docker-compose exec speech-recognition bash
    goto :eof
)

if "%1"=="clean" (
    echo Cleaning up Docker containers and images...
    docker-compose down --rmi all --volumes --remove-orphans
    goto :eof
)

echo Speech Recognition Docker Management
echo Usage: %0 {build^|start^|stop^|restart^|logs^|train^|shell^|clean}
echo.
echo Commands:
echo   build   - Build the Docker image
echo   start   - Start the application
echo   stop    - Stop the application
echo   restart - Restart the application
echo   logs    - Show application logs
echo   train   - Run training in container
echo   shell   - Open bash shell in container
echo   clean   - Clean up all Docker resources
echo.
echo Configuration:
echo   Edit .env file to switch between development/production mode
echo   Comment/uncomment source volume mounts in docker-compose.yml for production
