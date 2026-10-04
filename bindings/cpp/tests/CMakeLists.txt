cmake_minimum_required(VERSION 3.12)
project(fastloess_tests CXX)

set(CMAKE_CXX_STANDARD 17)

# Paths
set(BINDINGS_DIR "${CMAKE_SOURCE_DIR}/../../bindings/cpp")
include_directories("${BINDINGS_DIR}/include")

set(FASTLOESS_LIB
    ""
    CACHE FILEPATH
    "Path to the compiled fastloess C++ library"
)
set(FASTLOESS_LIB_DIR
    "${CMAKE_SOURCE_DIR}/../../target/release-c"
    CACHE PATH
    "Directory containing the compiled fastloess C++ library"
)

if(FASTLOESS_LIB)
    get_filename_component(FASTLOESS_LIB "${FASTLOESS_LIB}" ABSOLUTE)
else()
    if(WIN32)
        if(MSVC)
            set(FASTLOESS_CANDIDATE_NAMES fastloess_cpp fastloess_cpp.dll.lib)
        else()
            set(FASTLOESS_CANDIDATE_NAMES
                fastloess_cpp
                libfastloess_cpp.dll.a
                libfastloess_cpp.a
            )
        endif()
    else()
        set(FASTLOESS_CANDIDATE_NAMES
            fastloess_cpp
            libfastloess_cpp.so
            libfastloess_cpp.a
        )
    endif()

    find_library(
        FASTLOESS_LIB_FOUND
        NAMES ${FASTLOESS_CANDIDATE_NAMES}
        PATHS "${BINDINGS_DIR}/build" "${FASTLOESS_LIB_DIR}"
        NO_DEFAULT_PATH
    )

    set(FASTLOESS_LIB "${FASTLOESS_LIB_FOUND}")
endif()

if(NOT FASTLOESS_LIB OR FASTLOESS_LIB MATCHES "-NOTFOUND$")
    message(
        FATAL_ERROR
        "Could not find libfastloess_cpp. Please build bindings/cpp first."
    )
endif()

add_executable(test_fastloess_suite test_fastloess.cpp)
if(WIN32)
    if(MSVC)
        # MSVC compiler - use Windows system libraries
        target_link_libraries(
            test_fastloess_suite
            ${FASTLOESS_LIB}
            ntdll
            ws2_32
            userenv
            bcrypt
        )
    else()
        # MinGW compiler - link against the GNU-compatible Rust artifacts.
        target_link_libraries(
            test_fastloess_suite
            ${FASTLOESS_LIB}
            ntdll
            ws2_32
            userenv
            bcrypt
            pthread
        )
        get_filename_component(MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        file(GLOB MINGW_RUNTIME_DLLS
            "${MINGW_BIN_DIR}/libgcc_s_*-1.dll"
            "${MINGW_BIN_DIR}/libstdc++-6.dll"
            "${MINGW_BIN_DIR}/libwinpthread-1.dll"
            "${MINGW_BIN_DIR}/libmcfgthread-*.dll"
        )
        foreach(MINGW_RUNTIME_DLL IN LISTS MINGW_RUNTIME_DLLS)
            add_custom_command(TARGET test_fastloess_suite POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MINGW_RUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:test_fastloess_suite>"
                VERBATIM
            )
        endforeach()
    endif()
else()
    target_link_libraries(test_fastloess_suite ${FASTLOESS_LIB} pthread dl)
endif()

# Add test
enable_testing()
add_test(NAME test_fastloess_suite COMMAND test_fastloess_suite)
