@echo off
REM Setup MSVC env and run cargo. Requires Visual Studio Build Tools with
REM C++ workload AND Windows SDK installed at default paths.

setlocal
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo MSVC vcvars64 not found or failed. Install VS Build Tools 2022 with C++ workload and Windows 10/11 SDK.
    exit /b 1
)
set "PATH=C:\Users\Jose Diaz\.cargo\bin;%PATH%"
cargo %*
