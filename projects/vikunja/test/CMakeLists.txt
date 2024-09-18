cmake_minimum_required(VERSION 3.22)

message(STATUS "enable vikunja tests")

add_library(vikunjaTest INTERFACE)
target_link_libraries(vikunjaTest 
    INTERFACE
    vikunja::vikunja
    # alpaka test target
    common
)
add_library(vikunja::test ALIAS vikunjaTest)

add_subdirectory(integ)
