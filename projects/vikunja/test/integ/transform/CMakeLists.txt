cmake_minimum_required(VERSION 3.22)

set(_TARGET_NAME vikunjaIntegTransform)

project(${_TARGET_NAME} LANGUAGES CXX)

alpaka_add_executable(
    ${_TARGET_NAME}
    src/transform.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC 
    vikunja::test
    )

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_vikunja_TEST_OPTIONS})
