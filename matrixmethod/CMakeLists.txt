cmake_minimum_required(VERSION 3.3)
project(matrixmethod)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    matrixmethod.cpp)

add_executable(matrixmethod ${SOURCE_FILES})