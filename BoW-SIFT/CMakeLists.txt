cmake_minimum_required(VERSION 3.6)
project(BoW_SIFT)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV 2.4.3 REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(BoW_SIFT ${SOURCE_FILES})

target_link_libraries(BoW_SIFT ${OpenCV_LIBS} )