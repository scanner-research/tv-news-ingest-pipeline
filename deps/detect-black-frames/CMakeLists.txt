cmake_minimum_required(VERSION 2.8)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project( detect_black_frames )
find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( detect_black_frames detect_black_frames.cpp )
target_link_libraries( detect_black_frames ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} )


