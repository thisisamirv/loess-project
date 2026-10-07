cmake_minimum_required(VERSION 3.24)
project(fastloess-vcpkg-consumer LANGUAGES CXX)

find_package(unofficial-fastloess CONFIG REQUIRED)
add_executable(fastloess-consumer main.cpp)
target_link_libraries(fastloess-consumer PRIVATE unofficial::fastloess::fastloess)
add_custom_command(
    TARGET fastloess-consumer
    POST_BUILD
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:unofficial::fastloess::fastloess>"
        "$<TARGET_FILE_DIR:fastloess-consumer>"
)
enable_testing()
add_test(NAME fastloess-consumer COMMAND fastloess-consumer)
