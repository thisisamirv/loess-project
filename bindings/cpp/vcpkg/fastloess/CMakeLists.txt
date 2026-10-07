cmake_minimum_required(VERSION 3.24)
project(fastloess-vcpkg VERSION 3.0.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
get_filename_component(FASTLOESS_BINARY_NAME "${FASTLOESS_BINARY}" NAME)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This prebuilt package requires a 64-bit target ABI.")
endif()

if(WIN32)
    if(NOT MSVC)
        message(FATAL_ERROR "Prebuilt Windows packages require the MSVC ABI.")
    endif()
    if(FASTLOESS_ARCH STREQUAL "")
        set(FASTLOESS_ARCH x64)
    endif()
    if(FASTLOESS_ARCH STREQUAL "x64")
        set(import_machine x64)
    elseif(FASTLOESS_ARCH STREQUAL "arm64")
        set(import_machine ARM64)
    else()
        message(
            FATAL_ERROR
            "Unsupported Windows architecture: ${FASTLOESS_ARCH}"
        )
    endif()

    set(import_library "${CMAKE_CURRENT_BINARY_DIR}/fastloess_cpp.dll.lib")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/fastloess.def" import_definition)
    string(
        REPLACE "fastloess-win32-x64.dll"
        "${FASTLOESS_BINARY_NAME}"
        import_definition
        "${import_definition}"
    )
    file(
        WRITE "${CMAKE_CURRENT_BINARY_DIR}/fastloess.def"
        "${import_definition}"
    )
    add_custom_command(
        OUTPUT "${import_library}"
        COMMAND
            "${CMAKE_AR}" /nologo "/machine:${import_machine}"
            "/def:${CMAKE_CURRENT_BINARY_DIR}/fastloess.def"
            "/out:${import_library}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/fastloess.def"
        VERBATIM
    )
    add_custom_target(fastloess-import-library ALL DEPENDS "${import_library}")

    install(FILES "${FASTLOESS_BINARY}" DESTINATION bin)
    install(FILES "${import_library}" DESTINATION lib)
else()
    install(FILES "${FASTLOESS_BINARY}" DESTINATION lib)
endif()
install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/fastloess.h"
        "${FASTLOESS_SOURCE_DIR}/bindings/cpp/include/fastloess.hpp"
    DESTINATION include
)
if(EXISTS "${FASTLOESS_SOURCE_DIR}/bindings/cpp/include/fastloess_version.h")
    install(
        FILES "${FASTLOESS_SOURCE_DIR}/bindings/cpp/include/fastloess_version.h"
        DESTINATION include
    )
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/fastloessConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfig.cmake"
    INSTALL_DESTINATION share/fastloess
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/fastloessConfigVersion.cmake"
    DESTINATION share/fastloess
)
