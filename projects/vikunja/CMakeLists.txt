cmake_minimum_required(VERSION 3.22)

message(STATUS "enable vikunja")

# check requirements

if(alpaka_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "vikunja requires C++20 or later")
endif()

if(alpaka_USE_MDSPAN STREQUAL "OFF")
    message(FATAL_ERROR "vikunja requires mdspan")
endif()

if(NOT TARGET alpaka::alpaka)
    message(FATAL_ERROR "target alpaka::alpaka not defined")
endif()

# options

option(vikunja_BUILD_EXAMPLES "enable vikunja examples" OFF)

# vikunja target

add_library(vikunja INTERFACE)
target_include_directories(vikunja INTERFACE include)
target_link_libraries(vikunja INTERFACE alpaka::alpaka)
add_library(vikunja::vikunja ALIAS vikunja)

# enable example and tests

if(vikunja_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
