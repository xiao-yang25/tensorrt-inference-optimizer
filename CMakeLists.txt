cmake_minimum_required(VERSION 3.18)
project(tensorrt_inference_optimizer LANGUAGES CXX CUDA)

# Avoid generating `-isystem /usr/include` from imported targets (yaml-cpp on Ubuntu),
# which can break libstdc++ include_next chain in some environments.
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(TIO_ENABLE_BENCHMARK "Build benchmark binary" OFF)
option(TIO_ENABLE_PLUGIN "Build custom plugins" OFF)
option(TIO_ENABLE_TESTS "Register and run CTest tests" ON)
option(TIO_STRICT_TARGETS "Fail when optional sources are missing" OFF)

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
  ${PROJECT_SOURCE_DIR}/plugin/include
)

function(tio_apply_target_includes target visibility)
  target_include_directories(${target} ${visibility} ${PROJECT_INCLUDE_DIRS})
  if(NOT TENSORRT_INCLUDE_DIR STREQUAL "/usr/include" AND
     NOT TENSORRT_INCLUDE_DIR STREQUAL "/usr/include/x86_64-linux-gnu")
    target_include_directories(${target} SYSTEM ${visibility} ${TENSORRT_INCLUDE_DIR})
  endif()
endfunction()

add_library(tio_core
  src/config.cpp
  engine/builder.cpp
  engine/calibrator.cpp
  runtime/buffer_manager.cpp
  runtime/infer.cpp
  runtime/bevpool_cuda.cu
  runtime/bev_align_cuda.cu
  runtime/two_stage_pipeline.cpp
)

tio_apply_target_includes(tio_core PUBLIC)
target_link_libraries(tio_core PUBLIC
  yaml-cpp
  CUDA::cudart
  ${TENSORRT_LIB_NVINFER}
  ${TENSORRT_LIB_NVPARSER}
)

if(TIO_ENABLE_PLUGIN)
  set(TIO_PLUGIN_SOURCES
    plugin/src/alignbev_plugin.cpp
    plugin/cuda/alignbev_plugin_kernel.cu
    plugin/src/bevpool_plugin.cpp
    plugin/cuda/bevpool_plugin_kernel.cu
    plugin/src/gatherbev_plugin.cpp
    plugin/cuda/gatherbev_plugin_kernel.cu
    plugin/src/plugin_registry.cpp
  )
  set(TIO_PLUGIN_OK TRUE)
  foreach(SRC ${TIO_PLUGIN_SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
      set(TIO_PLUGIN_OK FALSE)
    endif()
  endforeach()
  if(TIO_PLUGIN_OK)
    add_library(tio_plugins SHARED ${TIO_PLUGIN_SOURCES})
    tio_apply_target_includes(tio_plugins PUBLIC)
    target_link_libraries(tio_plugins PUBLIC
      CUDA::cudart
      ${TENSORRT_LIB_NVINFER}
    )
    set_target_properties(tio_plugins PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/tests/bevpool_kernel_test.cu")
      add_executable(tio_bevpool_kernel_test
        plugin/tests/bevpool_kernel_test.cu
        plugin/cuda/bevpool_plugin_kernel.cu
      )
      tio_apply_target_includes(tio_bevpool_kernel_test PRIVATE)
      target_link_libraries(tio_bevpool_kernel_test PRIVATE CUDA::cudart)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/tests/alignbev_kernel_test.cu")
      add_executable(tio_alignbev_kernel_test
        plugin/tests/alignbev_kernel_test.cu
        plugin/cuda/alignbev_plugin_kernel.cu
      )
      tio_apply_target_includes(tio_alignbev_kernel_test PRIVATE)
      target_link_libraries(tio_alignbev_kernel_test PRIVATE CUDA::cudart)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/tests/gatherbev_kernel_test.cu")
      add_executable(tio_gatherbev_kernel_test
        plugin/tests/gatherbev_kernel_test.cu
        plugin/cuda/gatherbev_plugin_kernel.cu
      )
      tio_apply_target_includes(tio_gatherbev_kernel_test PRIVATE)
      target_link_libraries(tio_gatherbev_kernel_test PRIVATE CUDA::cudart)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/tests/plugin_enqueue_smoke_test.cpp")
      add_executable(tio_plugin_enqueue_smoke_test
        plugin/tests/plugin_enqueue_smoke_test.cpp
        plugin/src/bevpool_plugin.cpp
        plugin/cuda/bevpool_plugin_kernel.cu
        plugin/src/alignbev_plugin.cpp
        plugin/cuda/alignbev_plugin_kernel.cu
        plugin/src/gatherbev_plugin.cpp
        plugin/cuda/gatherbev_plugin_kernel.cu
      )
      tio_apply_target_includes(tio_plugin_enqueue_smoke_test PRIVATE)
      target_link_libraries(tio_plugin_enqueue_smoke_test PRIVATE CUDA::cudart ${TENSORRT_LIB_NVINFER})
      set_target_properties(tio_plugin_enqueue_smoke_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/tests/plugin_minimal_trt_test.cpp")
      add_executable(tio_plugin_minimal_trt_test
        plugin/tests/plugin_minimal_trt_test.cpp
        plugin/src/bevpool_plugin.cpp
        plugin/cuda/bevpool_plugin_kernel.cu
        plugin/src/alignbev_plugin.cpp
        plugin/cuda/alignbev_plugin_kernel.cu
        plugin/src/gatherbev_plugin.cpp
        plugin/cuda/gatherbev_plugin_kernel.cu
        plugin/src/plugin_registry.cpp
      )
      tio_apply_target_includes(tio_plugin_minimal_trt_test PRIVATE)
      target_link_libraries(tio_plugin_minimal_trt_test PRIVATE CUDA::cudart ${TENSORRT_LIB_NVINFER})
      set_target_properties(tio_plugin_minimal_trt_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin/tests/plugin_chain_trt_test.cpp")
      add_executable(tio_plugin_chain_trt_test
        plugin/tests/plugin_chain_trt_test.cpp
        plugin/src/bevpool_plugin.cpp
        plugin/cuda/bevpool_plugin_kernel.cu
        plugin/src/alignbev_plugin.cpp
        plugin/cuda/alignbev_plugin_kernel.cu
        plugin/src/gatherbev_plugin.cpp
        plugin/cuda/gatherbev_plugin_kernel.cu
        plugin/src/plugin_registry.cpp
      )
      tio_apply_target_includes(tio_plugin_chain_trt_test PRIVATE)
      target_link_libraries(tio_plugin_chain_trt_test PRIVATE CUDA::cudart ${TENSORRT_LIB_NVINFER})
      set_target_properties(tio_plugin_chain_trt_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
  elseif(TIO_STRICT_TARGETS)
    message(FATAL_ERROR "Plugin sources are missing but TIO_ENABLE_PLUGIN=ON.")
  else()
    message(WARNING "Plugin sources missing; skip tio_plugins target.")
  endif()
endif()

add_executable(tio_demo src/demo_main.cpp)
tio_apply_target_includes(tio_demo PRIVATE)
target_link_libraries(tio_demo PRIVATE tio_core)

if(TIO_ENABLE_BENCHMARK)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/benchmark.cpp")
    add_executable(tio_benchmark bench/benchmark.cpp)
    tio_apply_target_includes(tio_benchmark PRIVATE)
    target_link_libraries(tio_benchmark PRIVATE tio_core)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/compare_engines.cpp")
      add_executable(tio_compare_engines bench/compare_engines.cpp)
      tio_apply_target_includes(tio_compare_engines PRIVATE)
      target_link_libraries(tio_compare_engines PRIVATE tio_core)
    endif()
  elseif(TIO_STRICT_TARGETS)
    message(FATAL_ERROR "bench/benchmark.cpp missing but TIO_ENABLE_BENCHMARK=ON.")
  else()
    message(WARNING "Benchmark source missing; skip tio_benchmark target.")
  endif()
endif()

if(TIO_ENABLE_TESTS)
  include(CTest)
  if(TIO_ENABLE_PLUGIN)
    if(TARGET tio_bevpool_kernel_test)
      add_test(NAME plugin_bevpool_kernel COMMAND $<TARGET_FILE:tio_bevpool_kernel_test>)
    endif()
    if(TARGET tio_alignbev_kernel_test)
      add_test(NAME plugin_alignbev_kernel COMMAND $<TARGET_FILE:tio_alignbev_kernel_test>)
    endif()
    if(TARGET tio_gatherbev_kernel_test)
      add_test(NAME plugin_gatherbev_kernel COMMAND $<TARGET_FILE:tio_gatherbev_kernel_test>)
    endif()
    if(TARGET tio_plugin_enqueue_smoke_test)
      add_test(NAME plugin_enqueue_smoke COMMAND $<TARGET_FILE:tio_plugin_enqueue_smoke_test>)
    endif()
    if(TARGET tio_plugin_minimal_trt_test)
      add_test(NAME plugin_minimal_trt COMMAND $<TARGET_FILE:tio_plugin_minimal_trt_test>)
    endif()
    if(TARGET tio_plugin_chain_trt_test)
      add_test(NAME plugin_chain_trt COMMAND $<TARGET_FILE:tio_plugin_chain_trt_test>)
    endif()
  endif()
endif()
