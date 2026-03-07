@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ====== Config ======
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "BUILD_ROOT=%PROJECT_ROOT%\build"
set "RELEASE_BUILD_DIR=%BUILD_ROOT%\release"
set "DEBUG_BUILD_DIR=%BUILD_ROOT%\debug"
set "OUT=blockgame.exe"
set "DEFAULT_JOBS=2"
set "VSDEVCMD="
set "CMAKE_EXE="
set "NINJA_EXE="
set "PARALLEL_JOBS="

rem ====== Help ======
if "%~1"=="" goto :usage
if "%~1"=="/?" goto :usage
if /i "%~1"=="help" goto :usage

rem ====== Commands ======
if /i "%~1"=="release" goto :release
if /i "%~1"=="debug"   goto :debug
if /i "%~1"=="run"     goto :run
if /i "%~1"=="clean"   goto :clean

echo Unknown command: %~1
goto :usage

:find_vsdevcmd
if defined VSDEVCMD if exist "%VSDEVCMD%" exit /b 0
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%~I\Common7\Tools\VsDevCmd.bat" (
            set "VSDEVCMD=%%~I\Common7\Tools\VsDevCmd.bat"
        ) else if exist "%%~I\Common7\Tools\LaunchDevCmd.bat" (
            set "VSDEVCMD=%%~I\Common7\Tools\LaunchDevCmd.bat"
        )
    )
)
if defined VSDEVCMD if exist "%VSDEVCMD%" exit /b 0
echo Could not find Visual Studio developer command prompt tooling.
echo Install Visual Studio Build Tools with the C++ toolchain or set VSDEVCMD manually.
exit /b 1

:setup_env
call :find_vsdevcmd || exit /b 1
call "%VSDEVCMD%" -arch=x64 >nul || exit /b 1
pushd "%PROJECT_ROOT%" >nul
exit /b 0

:teardown_env
popd >nul 2>nul
exit /b 0

:find_cmake
if defined CMAKE_EXE if exist "%CMAKE_EXE%" exit /b 0
for /f "delims=" %%I in ('where cmake 2^>nul') do (
    set "CMAKE_EXE=%%~fI"
    goto :find_cmake_done
)
if defined VSDEVCMD (
    for %%I in ("%VSDEVCMD%") do set "VS_TOOLS_DIR=%%~dpI"
    for %%I in ("!VS_TOOLS_DIR!..\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe") do (
        if exist "%%~fI" set "CMAKE_EXE=%%~fI"
    )
)
:find_cmake_done
if defined CMAKE_EXE if exist "%CMAKE_EXE%" exit /b 0
echo Could not find cmake.exe.
exit /b 1

:find_ninja
if defined NINJA_EXE if exist "%NINJA_EXE%" exit /b 0
for /f "delims=" %%I in ('where ninja 2^>nul') do (
    set "NINJA_EXE=%%~fI"
    goto :find_ninja_done
)
if defined VSDEVCMD (
    for %%I in ("%VSDEVCMD%") do set "VS_TOOLS_DIR=%%~dpI"
    for %%I in ("!VS_TOOLS_DIR!..\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe") do (
        if exist "%%~fI" set "NINJA_EXE=%%~fI"
    )
)
:find_ninja_done
if defined NINJA_EXE if exist "%NINJA_EXE%" exit /b 0
echo Could not find ninja.exe.
exit /b 1

:resolve_jobs
set "PARALLEL_JOBS=%BLOCKGAME_JOBS%"
if not defined PARALLEL_JOBS set "PARALLEL_JOBS=%DEFAULT_JOBS%"
if not defined PARALLEL_JOBS set "PARALLEL_JOBS=1"
if "%PARALLEL_JOBS%"=="0" set "PARALLEL_JOBS=1"
exit /b 0

:ensure_build_root
if not exist "%BUILD_ROOT%" mkdir "%BUILD_ROOT%" >nul 2>nul
exit /b 0

:clear_stale_ninja_state
set "TARGET_BUILD_DIR=%~1"
if exist "%TARGET_BUILD_DIR%\.ninja_lock" del /q "%TARGET_BUILD_DIR%\.ninja_lock" >nul 2>nul
if exist "%TARGET_BUILD_DIR%\.ninja_log.restat" del /q "%TARGET_BUILD_DIR%\.ninja_log.restat" >nul 2>nul
exit /b 0

:do_build
set "TARGET_BUILD_DIR=%~1"
set "TARGET_BUILD_TYPE=%~2"
call :setup_env || exit /b 1
call :find_cmake || (
    set "ERR=%ERRORLEVEL%"
    call :teardown_env
    exit /b %ERR%
)
call :find_ninja || (
    set "ERR=%ERRORLEVEL%"
    call :teardown_env
    exit /b %ERR%
)
call :resolve_jobs
echo.
echo Using CMake: %CMAKE_EXE%
echo Using Ninja: %NINJA_EXE%
echo Parallel jobs: %PARALLEL_JOBS%
echo Build dir: %TARGET_BUILD_DIR%
call :ensure_build_root || (
    set "ERR=%ERRORLEVEL%"
    call :teardown_env
    exit /b %ERR%
)
if not exist "%TARGET_BUILD_DIR%" mkdir "%TARGET_BUILD_DIR%" >nul 2>nul
call :clear_stale_ninja_state "%TARGET_BUILD_DIR%"
"%CMAKE_EXE%" -S "%PROJECT_ROOT%" -B "%TARGET_BUILD_DIR%" -G Ninja ^
  "-DCMAKE_MAKE_PROGRAM:FILEPATH=%NINJA_EXE%" ^
  "-DCMAKE_BUILD_TYPE:STRING=%TARGET_BUILD_TYPE%"
if errorlevel 1 (
    set "ERR=%ERRORLEVEL%"
    call :teardown_env
    exit /b %ERR%
)
"%CMAKE_EXE%" --build "%TARGET_BUILD_DIR%" --target blockgame --parallel %PARALLEL_JOBS%
if errorlevel 1 (
    set "ERR=%ERRORLEVEL%"
    call :teardown_env
    exit /b %ERR%
)
call :teardown_env
exit /b 0

:release
echo.
echo === Building Release x64 with CMake + Ninja ===
call :do_build "%RELEASE_BUILD_DIR%" Release
exit /b %ERRORLEVEL%

:debug
echo.
echo === Building Debug x64 with CMake + Ninja ===
call :do_build "%DEBUG_BUILD_DIR%" Debug
exit /b %ERRORLEVEL%

:run
echo.
echo === Build then Run (Release) ===
call "%~f0" release || exit /b 1
if not exist "%RELEASE_BUILD_DIR%\%OUT%" (
    echo Could not find %RELEASE_BUILD_DIR%\%OUT% after building.
    exit /b 1
)
echo.
echo === Launching %OUT% ===
"%RELEASE_BUILD_DIR%\%OUT%"
exit /b %ERRORLEVEL%

:clean
echo.
echo === Cleaning build artifacts ===
if exist "%BUILD_ROOT%" rd /s /q "%BUILD_ROOT%" 2>nul
del /q "%PROJECT_ROOT%\blockgame.exe" 2>nul
del /q "%PROJECT_ROOT%\blockgame.pdb" 2>nul
del /q "%PROJECT_ROOT%\glfw3.dll" 2>nul
echo Done.
exit /b 0

:usage
echo.
echo Usage:
echo   build_blockgame.bat release   Configure and build Release with CMake + Ninja
echo   build_blockgame.bat debug     Configure and build Debug with CMake + Ninja
echo   build_blockgame.bat run       Build Release then run the app
echo   build_blockgame.bat clean     Delete build outputs
echo.
echo Notes:
echo   - This script no longer uses the legacy one-shot CL release build.
echo   - Release builds default to %DEFAULT_JOBS% parallel jobs to keep memory usage controlled.
echo   - Override jobs if needed with: set BLOCKGAME_JOBS=1 ^& build_blockgame.bat release
exit /b 1
