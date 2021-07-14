cmake_minimum_required(VERSION 3.10.2)
project(3ddfa_v2)
option(DEBUG "build tests or not" OFF)

IF(DEBUG)
    message("build for debug")
    SET(CMAKE_CXX_FLAGS "-Wall -g -fPIC")
else()
    SET(CMAKE_CXX_FLAGS "-Wall -O3 -g3 -fPIC")
ENDIF()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_definitions(-fPIC)
find_package(Threads REQUIRED)
set(LIB_DIR ../third_party/MNN/libs)
set(INC_DIR ../third_party/MNN/include ../third_party/imageHelper)

MESSAGE("lib dir:" ${LIB_DIR})
MESSAGE("INC_DIR:" ${INC_DIR})

FILE(GLOB SRC ./*.cpp )

include_directories(
    # ${OpenCV_INCLUDE_DIRS}
    ${INC_DIR}
    ./
)

LINK_DIRECTORIES(
    ${LIB_DIR}
)

add_executable(mnn_demo ${SRC})
target_link_libraries(mnn_demo PRIVATE  ${CMAKE_THREAD_LIBS_INIT}  MNN)