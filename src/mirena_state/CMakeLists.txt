cmake_minimum_required(VERSION 3.8)
project(mirena_state)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wdeprecated-declarations)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mirena_common REQUIRED)

# Define explicit source files instead of using GLOB to avoid compilation duplication
set(SOURCES
  src/mirena_state.cpp
  src/ekf.cpp
)

add_executable(mirena_state ${SOURCES})

# Target-specific include directories
target_include_directories(mirena_state PRIVATE
  "src/"
  ${EIGEN3_INCLUDE_DIR}
)

# ROS 2 dependency mapping
ament_target_dependencies(mirena_state
  rclcpp
  tf2
  tf2_ros
  sensor_msgs
  nav_msgs
  geometry_msgs
  mirena_common
)

# Install directives
install(TARGETS mirena_state DESTINATION lib/${PROJECT_NAME})

# Register package layout
ament_package()
