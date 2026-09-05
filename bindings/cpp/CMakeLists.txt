cmake_minimum_required(VERSION 3.15)
project(fastloess-cpp VERSION 2.0.0 LANGUAGES CXX)

include(FetchContent)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Import Corrosion for Rust integration
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.6.1
)
FetchContent_MakeAvailable(Corrosion)

# Import the Rust crate
corrosion_import_crate(MANIFEST_PATH Cargo.toml)

# The target name is derived from [package] name in Cargo.toml ("fastloess-cpp")

set_target_properties(fastloess_cpp PROPERTIES EXPORT_NAME fastloess)

# Configure include directories
# This ensures that #include "fastloess.hpp" works for consumers
target_include_directories(
    fastloess_cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Create an alias for idiomatic usage (fastloess::fastloess)
add_library(fastloess::fastloess ALIAS fastloess_cpp)

# Installation configuration
set(fastloess_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/fastloess")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fastloessConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfig.cmake"
    INSTALL_DESTINATION "${fastloess_install_cmakedir}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    TARGETS fastloess_cpp
    EXPORT fastloessTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

export(
    EXPORT fastloessTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/fastloessTargets.cmake"
    NAMESPACE fastloess::
)

install(
    EXPORT fastloessTargets
    FILE fastloessTargets.cmake
    NAMESPACE fastloess::
    DESTINATION "${fastloess_install_cmakedir}"
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfigVersion.cmake"
    DESTINATION "${fastloess_install_cmakedir}"
)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE fastloess)

# Tests
enable_testing()
add_executable(test_fastloess tests/test_fastloess.cpp)
target_link_libraries(test_fastloess PRIVATE fastloess_cpp)
target_include_directories(
    test_fastloess
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
add_test(NAME test_fastloess COMMAND test_fastloess)
