cmake_minimum_required(VERSION 3.18)
project(realsr_edge_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# ONNX Runtime
# -----------------------------------------------------------------------------
# Point this at the extracted onnxruntime release tree (the directory that
# contains 'include/' and 'lib/'). On Windows you typically get this by
# downloading e.g. onnxruntime-win-x64-1.20.x.zip from the GitHub release
# page and extracting it. See cpp_inference/README.md for details.

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to extracted ONNX Runtime release directory")
if(NOT ONNXRUNTIME_DIR OR NOT EXISTS "${ONNXRUNTIME_DIR}/include")
    message(FATAL_ERROR
        "Please pass -DONNXRUNTIME_DIR=<path/to/onnxruntime-release> to cmake. "
        "The directory must contain 'include/' and 'lib/'. "
        "See cpp_inference/README.md.")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_DIR}")

# -----------------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------------
add_executable(sr_cli src/main.cpp)

target_include_directories(sr_cli PRIVATE
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/stb
)

# Find the import lib (Windows) or .so (Linux) inside the release tree
if(WIN32)
    target_link_libraries(sr_cli PRIVATE
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    )
    # On Windows the runtime DLL must sit next to the .exe at runtime;
    # copy it into the build output dir so users don't have to set PATH.
    add_custom_command(TARGET sr_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:sr_cli>/onnxruntime.dll"
        COMMENT "Copying onnxruntime.dll next to sr_cli.exe"
    )
else()
    target_link_libraries(sr_cli PRIVATE
        "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    )
endif()

# -----------------------------------------------------------------------------
# Compiler warnings (sane defaults; not required for correctness)
# -----------------------------------------------------------------------------
if(MSVC)
    target_compile_options(sr_cli PRIVATE /W4)
else()
    target_compile_options(sr_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()
