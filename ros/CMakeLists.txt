cmake_minimum_required(VERSION 3.0)

project(mantrap)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

# Get the information about this package's buildtime dependencies.
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs)

# Declare that this catkin package's runtime dependencies.
catkin_package()
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(talker nodes/talker.cpp)
target_link_libraries(talker ${catkin_LIBRARIES})
