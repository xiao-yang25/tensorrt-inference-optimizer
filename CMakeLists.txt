cmake_minimum_required(VERSION 3.18)
project(tensorrt_inference_optimizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(TIO_ENABLE_BENCHMARK "Build benchmark binary" ON)
option(TIO_ENABLE_PLUGIN "Build custom plugins" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(yaml-cpp REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    /usr/include
    /usr/local/include
    /usr/include/x86_64-linux-gnu
    ${TensorRT_ROOT}/include
    $ENV{TENSORRT_ROOT}/include
)

find_library(TENSORRT_LIB_NVINFER nvinfer
  HINTS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    ${TensorRT_ROOT}/lib
    $ENV{TENSORRT_ROOT}/lib
)

find_library(TENSORRT_LIB_NVPARSER nvonnxparser
  HINTS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    ${TensorRT_ROOT}/lib
    $ENV{TENSORRT_ROOT}/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB_NVINFER OR NOT TENSORRT_LIB_NVPARSER)
  message(FATAL_ERROR "TensorRT not found. Set TensorRT_ROOT or TENSORRT_ROOT env.")
endif()

enable_language(CUDA)

set(PROJECT_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/engine
  ${PROJECT_SOURCE_DIR}/runtime
  ${PROJECT_SOURCE_DIR}/plugin
  ${TENSORRT_INCLUDE_DIR}
)

add_library(tio_core
  src/config.cpp
  engine/builder.cpp
  engine/calibrator.cpp
  runtime/buffer_manager.cpp
  runtime/infer.cpp
)

target_include_directories(tio_core PUBLIC ${PROJECT_INCLUDE_DIRS})
target_link_libraries(tio_core PUBLIC
  yaml-cpp
  CUDA::cudart
  ${TENSORRT_LIB_NVINFER}
  ${TENSORRT_LIB_NVPARSER}
)

if(TIO_ENABLE_PLUGIN)
  add_library(tio_plugins SHARED
    plugin/bevpool_plugin.cpp
    plugin/bevpool_plugin_kernel.cu
    plugin/plugin_registry.cpp
  )
  target_include_directories(tio_plugins PUBLIC ${PROJECT_INCLUDE_DIRS})
  target_link_libraries(tio_plugins PUBLIC
    CUDA::cudart
    ${TENSORRT_LIB_NVINFER}
  )
  set_target_properties(tio_plugins PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(tio_demo src/demo_main.cpp)
target_include_directories(tio_demo PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(tio_demo PRIVATE tio_core)

if(TIO_ENABLE_BENCHMARK)
  add_executable(tio_benchmark bench/benchmark.cpp)
  target_include_directories(tio_benchmark PRIVATE ${PROJECT_INCLUDE_DIRS})
  target_link_libraries(tio_benchmark PRIVATE tio_core)
endif()
