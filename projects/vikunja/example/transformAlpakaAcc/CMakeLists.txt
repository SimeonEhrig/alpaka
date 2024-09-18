cmake_minimum_required(VERSION 3.22)

set(_TARGET_NAME vikunjaTransformAlpakaAcc)

project(${_TARGET_NAME} LANGUAGES CXX)

alpaka_add_executable(
    ${_TARGET_NAME}
    src/transformAlpakaAcc.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC 
    vikunja::vikunja
    )
