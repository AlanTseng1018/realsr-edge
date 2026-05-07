@echo off
rem Run edsr_runner.exe with the ORT/CUDA/TRT runtime DLLs on PATH.
rem ORT GPU package's CUDA EP needs CUDA runtime DLLs (we use the ones bundled by torch).
rem TRT EP additionally needs TensorRT runtime DLLs (we use the ones bundled by tensorrt_libs).
setlocal

set ROOT=%~dp0..
set ORT_LIB=%ROOT%\third_party\onnxruntime\lib
set TORCH_LIB=%ROOT%\.venv\Lib\site-packages\torch\lib
set TRT_LIB=%ROOT%\.venv\Lib\site-packages\tensorrt_libs

set PATH=%ORT_LIB%;%TORCH_LIB%;%TRT_LIB%;%PATH%

"%~dp0build\edsr_runner.exe" %*
endlocal
