@echo off
rem Build edsr_runner.exe using VS 2022 BuildTools cl.exe + ORT GPU prebuilt headers/libs.
rem Run from Git Bash via:   cmd //c build.bat
rem Or from cmd.exe:         build.bat
setlocal

set ROOT=%~dp0..
set ORT=%ROOT%\third_party\onnxruntime
set STB=%ROOT%\third_party\stb
set OUT=%~dp0build

if not exist "%OUT%" mkdir "%OUT%"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64 failed & exit /b 1)

cl.exe ^
    /nologo /std:c++17 /O2 /EHsc /MD /W3 ^
    /Fo"%OUT%\\" /Fe"%OUT%\edsr_runner.exe" ^
    /I"%ORT%\include" /I"%STB%" ^
    "%~dp0edsr_runner.cpp" ^
    /link /LIBPATH:"%ORT%\lib" onnxruntime.lib

if errorlevel 1 (echo BUILD FAILED & exit /b 1)

rem Copy ORT runtime DLLs next to the exe so Windows DLL search picks them up
rem BEFORE C:\Windows\System32\onnxruntime.dll (Windows ML built-in, older version).
copy /y "%ORT%\lib\onnxruntime.dll" "%OUT%\" >nul
copy /y "%ORT%\lib\onnxruntime_providers_shared.dll" "%OUT%\" >nul
copy /y "%ORT%\lib\onnxruntime_providers_cuda.dll" "%OUT%\" >nul 2>&1
copy /y "%ORT%\lib\onnxruntime_providers_tensorrt.dll" "%OUT%\" >nul 2>&1

echo.
echo built: %OUT%\edsr_runner.exe
endlocal
