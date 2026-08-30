cmake_minimum_required(VERSION 3.16)
project(mirena_perception)

# The detector runs per cloud at sensor rate and is templated Eigen throughout.
# An unoptimised build does not keep up, so default to Release rather than
# inheriting CMake's empty build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(mirena_common REQUIRED)

# Only what the debug clouds need. The detector itself is Eigen and a grid; it
# has no k-d tree, no voxel grid and no clustering, so PCL is down to `common`
# and the conversion helpers, and nothing is vendored at configure time.
find_package(PCL 1.12 REQUIRED COMPONENTS common)
find_package(pcl_conversions REQUIRED)

add_executable(mirena_perception
  src/mirena_perception.cpp
  src/cone_detector.cpp
)

target_include_directories(mirena_perception PRIVATE
  "src/"
  ${EIGEN3_INCLUDE_DIR}
)

# SYSTEM so -Wpedantic below stays about our code: PCL's headers are not
# pedantic-clean and there are a lot of them.
target_include_directories(mirena_perception SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})

# Scoped to our target, not added globally: Patchwork++ is built as a
# subdirectory of this project and its warnings are not ours to fix.
target_compile_options(mirena_perception PRIVATE -Wall -Wextra -Wpedantic)

ament_target_dependencies(mirena_perception
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  pcl_conversions
  mirena_common
)

target_link_libraries(mirena_perception ${PCL_LIBRARIES})

install(TARGETS mirena_perception DESTINATION lib/${PROJECT_NAME})

ament_package()
