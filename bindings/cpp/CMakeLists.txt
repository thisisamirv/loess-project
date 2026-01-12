cmake_minimum_required(VERSION 3.15)
project(fastloess-cpp LANGUAGES CXX)

include(FetchContent)

# Import Corrosion for Rust integration
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.1
)
FetchContent_MakeAvailable(Corrosion)

# Import the Rust crate
corrosion_import_crate(MANIFEST_PATH Cargo.toml)

# The target name is derived from [package] name in Cargo.toml ("fastloess-cpp")

# Configure include directories
# This ensures that #include "fastloess.hpp" works for consumers
target_include_directories(fastloess_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create an alias for idiomatic usage (fastloess::fastloess)
add_library(fastloess::fastloess ALIAS fastloess_cpp)

# Installation configuration
include(GNUInstallDirs)

install(TARGETS fastloess_cpp
    EXPORT fastloessTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

export(EXPORT fastloessTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/fastloessTargets.cmake"
)
