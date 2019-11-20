cmake_minimum_required(VERSION 3.0)

project(mantrap_ros)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

## Project library installation.
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../include)
set(PROJECT_BIN_DIR ${PROJECT_SOURCE_DIR}/../bin)
set(PROJECT_LIB_DIR ${PROJECT_SOURCE_DIR}/../src)
set(PROJECT_EXTERN_DIR ${PROJECT_SOURCE_DIR}/../extern)
set(CPLUS_INCLUDE_PATH /usr/include)

# Package Library Installation.
include_directories(${CPLUS_INCLUDE_PATH})
include_directories(${EIGEN_INCLUDE_DIR})

include_directories(${PROJECT_INCLUDE_DIR})
file(GLOB gmm_stats_src ${PROJECT_LIB_DIR}/gmm_stats/*.cpp)

add_library(gmm_stats ${gmm_stats_src})
file(GLOB mantrap_src ${PROJECT_LIB_DIR}/mantrap/*/*.cpp)
add_library(mantrap ${mantrap_src})

## ROS Package Installation.
# Get the information about this package's buildtime dependencies.
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs nav_msgs message_generation)
# Generate package messages.
add_message_files(FILES Ado.msg Ego.msg)
generate_messages(DEPENDENCIES std_msgs geometry_msgs nav_msgs)
# Declare that this catkin package's runtime dependencies.
catkin_package(CATKIN_DEPENDS message_runtime)
include_directories(include ${catkin_INCLUDE_DIRS})

# Build nodes.
add_executable(env_publisher nodes/env_publisher.cpp)
target_link_libraries(env_publisher ${catkin_LIBRARIES} mantrap gmm_stats)
