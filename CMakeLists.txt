cmake_minimum_required(VERSION 3.0)
project(mantrap)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CPLUS_INCLUDE_PATH /usr/local/include)
set(LIBRARY_PATH /usr/local/lib)

# place binaries and libraries according to GNU standards
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(PROJECT_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

# external packages.
add_subdirectory(${PROJECT_LIB_DIR})

# internal packages.
include_directories(${PROJECT_INCLUDE_DIR})
add_subdirectory(src)

# testing.
enable_testing()
add_subdirectory(test)