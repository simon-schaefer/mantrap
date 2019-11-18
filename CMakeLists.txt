cmake_minimum_required(VERSION 3.0)
project(mantrap)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CPLUS_INCLUDE_PATH /usr/local/include)
set(LIBRARY_PATH /usr/local/lib)

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(PROJECT_LIB_DIR ${PROJECT_SOURCE_DIR}/src)
set(PROJECT_EXTERN_DIR ${PROJECT_SOURCE_DIR}/extern)

# external packages - eigen.
include_directories(${CPLUS_INCLUDE_PATH}/eigen3)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# internal packages.
include_directories(${PROJECT_INCLUDE_DIR})

file(GLOB gmm_stats_src ${PROJECT_LIB_DIR}/gmm_stats/gaussian2d.cpp ${PROJECT_LIB_DIR}/gmm_stats/gmm2d.cpp)
add_library(gmm_stats ${gmm_stats_src})
#file(GLOB mantrap_src ${PROJECT_LIB_DIR}/mantrap/*.cpp)
#add_library(mantrap ${mantrap_src})

# testing.
option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()

    # Add googletest submodule to project.
    add_subdirectory(${PROJECT_EXTERN_DIR}/googletest)

    # Exclude googletest from IDE.
    set_target_properties(gtest PROPERTIES FOLDER extern)
    set_target_properties(gtest_main PROPERTIES FOLDER extern)
    set_target_properties(gmock PROPERTIES FOLDER extern)
    set_target_properties(gmock_main PROPERTIES FOLDER extern)

    # Add test directory.
    add_subdirectory(test)
endif()