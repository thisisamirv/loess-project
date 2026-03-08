cmake_minimum_required(VERSION 3.12)
project(fastloess_tests CXX)

set(CMAKE_CXX_STANDARD 17)

# Paths
set(BINDINGS_DIR "${CMAKE_SOURCE_DIR}/../../bindings/cpp")
include_directories("${BINDINGS_DIR}/include")

# Find library (built by bindings/cpp/CMakeLists.txt)
# We assume the library is in bindings/cpp/build
find_library(FASTLOESS_LIB NAMES fastloess_cpp PATHS "${BINDINGS_DIR}/build" "${CMAKE_SOURCE_DIR}/../../target/release" NO_DEFAULT_PATH)

if (NOT FASTLOESS_LIB)
    message(FATAL_ERROR "Could not find libfastloess_cpp. Please build bindings/cpp first.")
endif()

add_executable(test_fastloess_suite test_fastloess.cpp)
target_link_libraries(test_fastloess_suite ${FASTLOESS_LIB} pthread dl)

# Add test
enable_testing()
add_test(NAME test_fastloess_suite COMMAND test_fastloess_suite)
