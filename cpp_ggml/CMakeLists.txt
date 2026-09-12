cmake_minimum_required(VERSION 3.14)
project(gkdgml LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options (mirrors the ultralytics-ggml / free-splatter-ggml integration style)
option(GKD_GGML_BUILD_CLI    "Build the gkd-cli binary"                 ON)
option(GKD_GGML_BUILD_TESTS  "Build the golden-op tests"                OFF)
option(GKD_GGML_SHARED       "Build libgkdgml as a shared library"      OFF)

# Backend forwarding to ggml. At most one GPU backend per build directory;
# CPU is always included (GPU builds fall back to CPU for unsupported ops).
option(GKD_GGML_CUDA    "Enable CUDA backend in ggml"    OFF)
option(GKD_GGML_METAL   "Enable Metal backend in ggml"   OFF)
option(GKD_GGML_VULKAN  "Enable Vulkan backend in ggml"  OFF)

set(_gkd_gpu_backends 0)
foreach(_backend GKD_GGML_CUDA GKD_GGML_METAL GKD_GGML_VULKAN)
    if(${_backend})
        math(EXPR _gkd_gpu_backends "${_gkd_gpu_backends} + 1")
    endif()
endforeach()
if(_gkd_gpu_backends GREATER 1)
    message(FATAL_ERROR "Select at most one GPU backend per build directory")
endif()

# ggml uses CMake's `native` CUDA architecture only with CMake >= 3.24. On older
# CMake, select the attached GPU instead of compiling a broad architecture list.
if(GKD_GGML_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_LESS 3.24)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _gkd_cuda_compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _gkd_cuda_compute_cap "${_gkd_cuda_compute_cap}")
        if(_gkd_cuda_compute_cap)
            string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${_gkd_cuda_compute_cap}")
            message(STATUS "CUDA architecture auto-detected: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
endif()

if(GKD_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Replay CUDA graphs" FORCE)
endif()
if(GKD_GGML_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(GKD_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml.
# GGML_NATIVE=ON adapts the CPU build to the host (AVX2/AVX-512); GGML_LLAMAFILE
# enables the tinyBLAS tiled SGEMM used by the F32 mul_mat hotspots.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE      ON  CACHE BOOL "Enable tinyBLAS SGEMM kernels in ggml" FORCE)

# ----------------------------------------------------------------------------
# ggml source patches.
#
# The submodule stays pristine at its pinned tag; every deviation from upstream
# lives in patches/ggml/*.patch and is applied automatically at configure time:
#   - forward check OK        -> not applied yet, apply now
#   - reverse check OK        -> already applied (idempotent re-configure), skip
#   - neither                 -> conflicting local edits, abort
# To extend the integration, drop a new `git -C third_party/ggml diff`-style
# patch into patches/ggml/ and re-run cmake.
# ----------------------------------------------------------------------------
find_package(Git QUIET)
set(GKD_GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
file(GLOB GKD_GGML_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml/*.patch)
list(SORT GKD_GGML_PATCHES)
if(GKD_GGML_PATCHES)
    if(NOT EXISTS ${GKD_GGML_DIR}/src)
        message(FATAL_ERROR "ggml submodule is missing; run: git submodule update --init --recursive")
    endif()
    if(NOT Git_FOUND)
        message(FATAL_ERROR "git is required to apply the bundled ggml patches")
    endif()
    foreach(_patch ${GKD_GGML_PATCHES})
        get_filename_component(_pname ${_patch} NAME)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${GKD_GGML_DIR} apply --check ${_patch}
            RESULT_VARIABLE _fwd
            OUTPUT_QUIET ERROR_QUIET)
        if(_fwd EQUAL 0)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} -C ${GKD_GGML_DIR} apply ${_patch}
                RESULT_VARIABLE _applied)
            if(NOT _applied EQUAL 0)
                message(FATAL_ERROR "failed to apply ggml patch: ${_pname}")
            endif()
            message(STATUS "ggml patch applied: ${_pname}")
            continue()
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${GKD_GGML_DIR} apply --reverse --check ${_patch}
            RESULT_VARIABLE _rev
            OUTPUT_QUIET ERROR_QUIET)
        if(_rev EQUAL 0)
            message(STATUS "ggml patch already applied: ${_pname}")
        else()
            message(FATAL_ERROR "ggml patch conflicts with the current submodule state: ${_pname}")
        endif()
    endforeach()
endif()

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# stb (header-only image codecs)
add_library(gkd_stb INTERFACE)
target_include_directories(gkd_stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

# zlib for the BPE vocab file (gz)
find_package(ZLIB QUIET)

set(GKDGML_SOURCES
    src/common.cpp
    src/backend.cpp
    src/gguf_loader.cpp
    src/tokenizer.cpp
    src/image_io.cpp
    src/gkd_graph.cpp
    src/postprocess.cpp
    src/capi.cpp
)

if(GKD_GGML_SHARED)
    add_library(gkdgml SHARED ${GKDGML_SOURCES})
else()
    add_library(gkdgml STATIC ${GKDGML_SOURCES})
endif()

if(GKD_GGML_CUDA)
    target_compile_definitions(gkdgml PRIVATE GKD_USE_CUDA)
endif()
if(GKD_GGML_METAL)
    target_compile_definitions(gkdgml PRIVATE GKD_USE_METAL)
endif()
if(GKD_GGML_VULKAN)
    target_compile_definitions(gkdgml PRIVATE GKD_USE_VULKAN)
endif()

target_include_directories(gkdgml
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gkdgml
    PUBLIC  ggml
    PRIVATE gkd_stb)
if(ZLIB_FOUND)
    target_compile_definitions(gkdgml PRIVATE GKD_HAVE_ZLIB)
    target_link_libraries(gkdgml PRIVATE ZLIB::ZLIB)
else()
    message(STATUS "zlib not found: pass an already-extracted BPE vocab file via --bpe-txt")
endif()

# Exact-JPEG decoding. The official pipeline reads images through cv2.imread,
# which uses libjpeg: stb's own JPEG decoder differs by up to +-3/255 on ~11%
# of pixels, which propagated through the network as preprocessing noise.
# Optional - the engine falls back to stb_image when libjpeg is unavailable.
find_package(JPEG QUIET)
if(JPEG_FOUND)
    target_compile_definitions(gkdgml PRIVATE GKD_HAVE_LIBJPEG)
    target_link_libraries(gkdgml PRIVATE ${JPEG_LIBRARIES})
    target_include_directories(gkdgml PRIVATE ${JPEG_INCLUDE_DIRS})
    message(STATUS "libjpeg found: JPEG inputs decode through libjpeg (cv2/PIL bit-exact)")
else()
    message(STATUS "libjpeg not found: JPEG inputs decode through stb_image (small pixel deviations)")
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gkdgml PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(gkdgml PRIVATE GKD_USE_OPENMP)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gkdgml PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# CLI
if(GKD_GGML_BUILD_CLI)
    add_executable(gkd-cli src/cli.cpp)
    target_link_libraries(gkd-cli PRIVATE gkdgml)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gkd-cli PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
endif()

# Tests
if(GKD_GGML_BUILD_TESTS)
    add_executable(gkd-test-ops tests/test_ops.cpp)
    target_link_libraries(gkd-test-ops PRIVATE gkdgml)
    # white-box test: includes src/tokenizer.cpp for the static ftfy subset
    add_executable(gkd-test-ftfy tests/test_ftfy.cpp src/common.cpp)  # test_ftfy includes tokenizer.cpp (white-box)
    target_link_libraries(gkd-test-ftfy PRIVATE ggml)
endif()
