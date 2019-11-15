cmake_minimum_required(VERSION 3.0)

project(murseco)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(MURSECO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(MURSECO_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(MURSECO_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

include_directories(${MURSECO_INCLUDE_DIR})
include_directories(${MURSECO_LIB_DIR})

# external packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# testing
enable_testing()

add_subdirectory(src)
add_subdirectory(test)

# install(DIRECTORY include/colormap DESTINATION include)
