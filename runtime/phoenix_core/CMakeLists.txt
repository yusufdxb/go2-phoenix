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

add_library(phoenix_core STATIC
  src/types.cpp
  src/attitude.cpp
  src/gate.cpp
  src/filters.cpp
)

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
    )
    target_link_libraries(test_phoenix_core PRIVATE phoenix_core GTest::gtest GTest::gtest_main)
    # Absolute path so the parity fixture resolves regardless of working dir.
    target_compile_definitions(test_phoenix_core PRIVATE
      PHOENIX_PARITY_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/parity_v1.txt")
    target_compile_options(test_phoenix_core PRIVATE -Wall -Wextra)
    add_test(NAME phoenix_core_unit COMMAND test_phoenix_core)
  else()
    message(STATUS "GTest not found; phoenix_core tests will not be built")
  endif()
endif()
