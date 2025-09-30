@echo off
setlocal enabledelayedexpansion

rem ====== Config ======
set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
set "PROJECT_ROOT=%~dp0"
set "INCLUDE_DIR=%PROJECT_ROOT%include"
set "LIB_DIR=%PROJECT_ROOT%libs"
set "SRC=src\main.cpp src\glad.c src\TextureLoader.cpp"
set "OUT=blockgame.exe"
set "GLFW_LIB=glfw3.lib"
set "GLFW_DLL=glfw3.dll"

rem ====== Help ======
if "%~1"=="" goto :usage
if "%~1"=="/?" goto :usage
if /i "%~1"=="help" goto :usage

rem ====== Ensure VS tools ======
if not exist "%VSDEVCMD%" (
  echo Could not find VsDevCmd.bat at:
  echo   %VSDEVCMD%
  echo Update VSDEVCMD in this script.
  exit /b 1
)

rem ====== Ensure dependencies ======
if not exist "%INCLUDE_DIR%" (
  echo Could not find include folder at:
  echo   %INCLUDE_DIR%
  echo Expected headers ^(glad, GLFW, glm, stb^).
  exit /b 1
)
if not exist "%LIB_DIR%\%GLFW_LIB%" (
  echo Could not find GLFW library at:
  echo   %LIB_DIR%\%GLFW_LIB%
  exit /b 1
)

set "HAS_GLFW_DLL="
if exist "%LIB_DIR%\%GLFW_DLL%" (
  set "HAS_GLFW_DLL=1"
) else (
  echo Note: %GLFW_DLL% not found in %LIB_DIR% - skipping runtime copy.
)

rem ====== Commands ======
if /i "%~1"=="release" goto :release
if /i "%~1"=="debug"   goto :debug
if /i "%~1"=="run"     goto :run
if /i "%~1"=="clean"   goto :clean

echo Unknown command: %~1
goto :usage

:setup_env
call "%VSDEVCMD%" -arch=x64 || exit /b 1
pushd "%PROJECT_ROOT%" >nul
exit /b 0

:teardown_env
popd >nul 2>nul
exit /b 0

:copy_runtime
if defined HAS_GLFW_DLL (
  copy /Y "%LIB_DIR%\%GLFW_DLL%" "%PROJECT_ROOT%%GLFW_DLL%" >nul
)
exit /b 0

:release
echo.
echo === Building Release x64 ===
call :setup_env || exit /b 1
cl /nologo /EHsc /std:c++20 /O2 /DNDEBUG /MD ^
  /I "%INCLUDE_DIR%" ^
  %SRC% ^
  /Fe:%OUT% ^
  /link /LIBPATH:"%LIB_DIR%" ^
  glfw3.lib opengl32.lib user32.lib gdi32.lib shell32.lib advapi32.lib
set "ERR=%ERRORLEVEL%"
call :teardown_env
if not "%ERR%"=="0" exit /b %ERR%
call :copy_runtime
exit /b %ERRORLEVEL%

:debug
echo.
echo === Building Debug x64 ===
call :setup_env || exit /b 1
cl /nologo /EHsc /std:c++20 /MDd /Zi /Od /DDEBUG ^
  /I "%INCLUDE_DIR%" ^
  %SRC% ^
  /Fe:%OUT% ^
  /link /LIBPATH:"%LIB_DIR%" ^
  glfw3.lib opengl32.lib user32.lib gdi32.lib shell32.lib advapi32.lib
set "ERR=%ERRORLEVEL%"
call :teardown_env
if not "%ERR%"=="0" exit /b %ERR%
call :copy_runtime
exit /b %ERRORLEVEL%

:run
echo.
echo === Build then Run (Release) ===
call "%~f0" release || exit /b 1
echo.
echo === Launching %OUT% ===
"%PROJECT_ROOT%%OUT%"
exit /b %ERRORLEVEL%

:clean
echo.
echo === Cleaning build artifacts ===
del /q "%PROJECT_ROOT%"*.obj 2>nul
del /q "%PROJECT_ROOT%"*.pdb 2>nul
del /q "%PROJECT_ROOT%"*.ilk 2>nul
del /q "%PROJECT_ROOT%%OUT%" 2>nul
if defined HAS_GLFW_DLL del /q "%PROJECT_ROOT%%GLFW_DLL%" 2>nul
echo Done.
exit /b 0

:usage
echo.
echo Usage:
echo   build_blockgame.bat release   Build release with MSVC and GLFW
echo   build_blockgame.bat debug     Build debug with MSVC and GLFW
echo   build_blockgame.bat run       Build release then run the app
echo   build_blockgame.bat clean     Delete build outputs
echo.
echo Edit VSDEVCMD at the top if your VS path differs.
exit /b 1
