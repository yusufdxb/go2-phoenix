cmake_minimum_required(VERSION 3.16)
project(phoenix_core LANGUAGES CXX)

# C++17, not C++20. ROS 2 Humble's default is 17, the Orin toolchain is the
# constraint that matters, and the only C++20 feature this design wanted was
# std::span. A fixed-size std::array carries the same contract here without a
# toolchain compatibility risk on the target.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PHOENIX_CORE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# ONNX Runtime is optional so the deterministic core stays buildable anywhere,
# including a CI runner with no model runtime installed. Point ONNXRUNTIME_ROOT
# at a prebuilt release directory (the one containing include/ and lib/).
# On the Jetson this must be an aarch64 build, not the x86_64 one.
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime install prefix")
if(NOT ONNXRUNTIME_ROOT AND EXISTS "$ENV{HOME}/.local/opt/onnxruntime/include/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_ROOT "$ENV{HOME}/.local/opt/onnxruntime")
endif()

set(PHOENIX_ORT_FOUND OFF)
if(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  set(PHOENIX_ORT_FOUND ON)
  message(STATUS "phoenix_core: ONNX Runtime at ${ONNXRUNTIME_ROOT}")
else()
  message(STATUS "phoenix_core: no ONNX Runtime; inference backend disabled")
endif()

add_library(phoenix_core STATIC
  src/types.cpp
  src/attitude.cpp
  src/gate.cpp
  src/filters.cpp
  src/shield.cpp
  src/ort_engine.cpp
)

if(PHOENIX_ORT_FOUND)
  target_compile_definitions(phoenix_core PUBLIC PHOENIX_WITH_ORT)
  target_include_directories(phoenix_core SYSTEM PUBLIC "${ONNXRUNTIME_ROOT}/include")
  target_link_libraries(phoenix_core PUBLIC "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  # rpath so the test and any consumer find the .so without LD_LIBRARY_PATH.
  set_target_properties(phoenix_core PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()

target_include_directories(phoenix_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_options(phoenix_core PRIVATE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
)

# No fast-math anywhere. The whole point of this library is bit-level agreement
# with a Python reference, and -ffast-math permits reassociation and assumes no
# NaN, which would silently break both the parity gates and the fail-closed
# NaN semantics the safety ladder depends on.
target_compile_options(phoenix_core PRIVATE -fno-fast-math)

if(PHOENIX_CORE_SANITIZE)
  target_compile_options(phoenix_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(phoenix_core PUBLIC -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(PHOENIX_CORE_BUILD_TESTS "Build phoenix_core unit tests" ON)

if(PHOENIX_CORE_BUILD_TESTS)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_phoenix_core
      test/test_attitude.cpp
      test/test_gate.cpp
      test/test_filters.cpp
      test/test_parity.cpp
      test/test_inference.cpp
      test/test_shield.cpp
    )
    target_link_libraries(test_phoenix_core PRIVATE phoenix_core GTest::gtest GTest::gtest_main)
    # Absolute path so the parity fixture resolves regardless of working dir.
    target_compile_definitions(test_phoenix_core PRIVATE
      PHOENIX_PARITY_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/parity_v1.txt"
      PHOENIX_DEPLOY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../deploy")
    if(PHOENIX_ORT_FOUND)
      set_target_properties(test_phoenix_core PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
    endif()
    target_compile_options(test_phoenix_core PRIVATE -Wall -Wextra)
    add_test(NAME phoenix_core_unit COMMAND test_phoenix_core)
  else()
    message(STATUS "GTest not found; phoenix_core tests will not be built")
  endif()
endif()
