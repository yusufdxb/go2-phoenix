cmake_minimum_required(VERSION 3.16)
project(phoenix_ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# phoenix_core is a plain CMake library, deliberately not an ament package: it
# must stay buildable and testable with no ROS installed at all.
add_subdirectory(../phoenix_core ${CMAKE_CURRENT_BINARY_DIR}/phoenix_core)

add_executable(policy_runtime_node src/policy_runtime_node.cpp)
target_link_libraries(policy_runtime_node phoenix_core)
ament_target_dependencies(policy_runtime_node
  rclcpp sensor_msgs std_msgs geometry_msgs)

install(TARGETS policy_runtime_node DESTINATION lib/${PROJECT_NAME})

ament_package()
