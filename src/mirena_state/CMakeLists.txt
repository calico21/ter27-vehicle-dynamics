cmake_minimum_required(VERSION 3.8)
project(mirena_state)

# Set C++17 standard for ROS 2 Jazzy and modern PCL compatibility
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wdeprecated-declarations)
endif()

# Global Eigen memory alignment fix to prevent PCL ABI crashes on ARM64
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=0)

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
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

# PCL compiler definitions
add_definitions(${PCL_DEFINITIONS})

# Source definitions
set(SOURCES
  src/mirena_state.cpp
  src/ekf.cpp
)

set(FAST_LIO_SOURCES
  src/fast_lio.cpp
)

# Executable declarations
add_executable(mirena_state_ekf ${SOURCES})
add_executable(mirena_state_fast_lio ${FAST_LIO_SOURCES})

# Target include directories
target_include_directories(mirena_state_ekf PRIVATE
  "src/"
  ${EIGEN3_INCLUDE_DIRS}
)

target_include_directories(mirena_state_fast_lio PRIVATE
  "src/"
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# ROS 2 dependency mapping
ament_target_dependencies(mirena_state_ekf
  rclcpp
  tf2
  tf2_ros
  sensor_msgs
  nav_msgs
  geometry_msgs
  mirena_common
)

ament_target_dependencies(mirena_state_fast_lio
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  pcl_conversions
  mirena_common
)

# External library linking
target_link_libraries(mirena_state_fast_lio
  ${PCL_LIBRARIES}
)

# Install directives
install(TARGETS mirena_state_ekf mirena_state_fast_lio DESTINATION lib/${PROJECT_NAME})

ament_package()