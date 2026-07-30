cmake_minimum_required(VERSION 3.8)
project(mirena_slam)

# GTSAM and Eigen are template-heavy: an unoptimised build is several times
# slower, which is the difference between hitting the keyframe budget and not.
# Default to Release rather than inheriting CMake's empty build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wdeprecated-declarations)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(GTSAM REQUIRED)
find_package(mirena_common REQUIRED)

# Define explicit source files instead of using GLOB to avoid compilation duplication
set(SOURCES
  src/mirena_slam.cpp
  src/slam.cpp
)

add_executable(mirena_slam ${SOURCES})

# Target-specific include directories
target_include_directories(mirena_slam PRIVATE 
  "src/"
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)

# ROS 2 dependency mapping
ament_target_dependencies(mirena_slam 
  rclcpp 
  rclcpp_lifecycle
  visualization_msgs 
  geometry_msgs
  mirena_common
  message_filters
  tf2_ros
  nav_msgs
)

# Link external libraries (GTSAM handles its own export targets best this way)
target_link_libraries(mirena_slam gtsam)

# Install directives
install(TARGETS mirena_slam DESTINATION lib/${PROJECT_NAME})

# Register package layout
ament_package()