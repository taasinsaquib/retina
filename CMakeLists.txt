# Specify the minimum version for CMake

cmake_minimum_required(VERSION 3.21)

# Project's name

project(hello)

find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(hello ${PROJECT_SOURCE_DIR}/simulation.cpp)
target_link_libraries(hello PRIVATE pybind11::embed)