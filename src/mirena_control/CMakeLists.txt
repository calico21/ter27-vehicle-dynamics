cmake_minimum_required(VERSION 3.16)
project(mirena_control CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MIRENA_CONTROL_NATIVE_ARCH
  "Tune the controller and the generated solver for the build machine" ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(mirena_common REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# acados ships as a plain CMake-less tree; locate it the same way its own
# Python/Matlab interfaces do.
if(DEFINED ENV{ACADOS_SOURCE_DIR})
  set(ACADOS_ROOT "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "acados installation root")
else()
  set(ACADOS_ROOT "$ENV{HOME}/acados" CACHE PATH "acados installation root")
endif()

if(NOT EXISTS "${ACADOS_ROOT}/include/acados_c/ocp_nlp_interface.h")
  message(FATAL_ERROR
    "acados headers not found under ${ACADOS_ROOT}. "
    "Set ACADOS_SOURCE_DIR or pass -DACADOS_ROOT=/path/to/acados.")
endif()

find_library(ACADOS_LIB acados PATHS "${ACADOS_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(HPIPM_LIB hpipm PATHS "${ACADOS_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(BLASFEO_LIB blasfeo PATHS "${ACADOS_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

set(ACADOS_INCLUDE_DIRS
  "${ACADOS_ROOT}/include"
  "${ACADOS_ROOT}/include/acados"
  "${ACADOS_ROOT}/include/blasfeo/include"
  "${ACADOS_ROOT}/include/hpipm/include"
)

# ---------------------------------------------------------------------------
# acados code generation
#
# The vehicle model and the OCP live in codegen/*.py, but only as a source
# language: acados turns them into C that is compiled into the executable
# below, so the running node never loads Python, CasADi or ctypes.
# Generation reruns whenever those scripts change.
# ---------------------------------------------------------------------------
set(SOLVER_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/acados_generated")
set(SOLVER_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/generate_solver.py")
set(SOLVER_MODEL_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/codegen/dbm_model.py")

file(SHA256 "${SOLVER_GEN_SCRIPT}" _gen_hash)
file(SHA256 "${SOLVER_MODEL_SCRIPT}" _model_hash)
set(SOLVER_GEN_STAMP_VALUE "${_gen_hash}-${_model_hash}")
set(SOLVER_GEN_STAMP "${SOLVER_GEN_DIR}/.codegen.stamp")

set(_regenerate TRUE)
if(EXISTS "${SOLVER_GEN_STAMP}" AND EXISTS "${SOLVER_GEN_DIR}/acados_solver_mpc_problem.h")
  file(READ "${SOLVER_GEN_STAMP}" _previous_stamp)
  if(_previous_stamp STREQUAL SOLVER_GEN_STAMP_VALUE)
    set(_regenerate FALSE)
  endif()
endif()

if(_regenerate)
  message(STATUS "mirena_control: generating the acados solver (this takes a moment)")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "ACADOS_SOURCE_DIR=${ACADOS_ROOT}"
            ${Python3_EXECUTABLE} "${SOLVER_GEN_SCRIPT}" --output "${SOLVER_GEN_DIR}"
    RESULT_VARIABLE _gen_result
    OUTPUT_VARIABLE _gen_output
    ERROR_VARIABLE _gen_error
  )
  if(NOT _gen_result EQUAL 0)
    message(FATAL_ERROR
      "acados code generation failed (exit ${_gen_result}).\n"
      "Requires python3 with acados_template and casadi.\n"
      "${_gen_output}\n${_gen_error}")
  endif()
  file(WRITE "${SOLVER_GEN_STAMP}" "${SOLVER_GEN_STAMP_VALUE}")
endif()

# Reconfigure if the generator changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${SOLVER_GEN_SCRIPT}" "${SOLVER_MODEL_SCRIPT}")

file(GLOB_RECURSE SOLVER_GEN_SOURCES "${SOLVER_GEN_DIR}/*.c")

# Remove the standalone test executables Acados generates, 
# otherwise their main() functions will conflict with your ROS node's main()
list(REMOVE_ITEM SOLVER_GEN_SOURCES
  "${SOLVER_GEN_DIR}/main_mpc_problem.c"
  "${SOLVER_GEN_DIR}/main_sim_mpc_problem.c"
)

add_library(mpc_solver_generated STATIC ${SOLVER_GEN_SOURCES})
target_include_directories(mpc_solver_generated PUBLIC
  "${SOLVER_GEN_DIR}" ${ACADOS_INCLUDE_DIRS})
target_link_libraries(mpc_solver_generated PUBLIC
  ${ACADOS_LIB} ${HPIPM_LIB} ${BLASFEO_LIB} m)
# CasADi's output is machine written; its warnings are not actionable.
target_compile_options(mpc_solver_generated PRIVATE -O3 -w)

# ---------------------------------------------------------------------------
# Controller library
# ---------------------------------------------------------------------------
add_library(mirena_control_core STATIC
  src/cubic_spline.cpp
  src/mpc_controller.cpp
)
target_include_directories(mirena_control_core PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(mirena_control_core PUBLIC mpc_solver_generated)
target_compile_options(mirena_control_core PRIVATE -O3 -Wall -Wextra -Wpedantic)

if(MIRENA_CONTROL_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native HAVE_MARCH_NATIVE)
  if(HAVE_MARCH_NATIVE)
    target_compile_options(mpc_solver_generated PRIVATE -march=native)
    target_compile_options(mirena_control_core PRIVATE -march=native)
  endif()
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(mirena_control src/mpc_node.cpp)
target_link_libraries(mirena_control mirena_control_core nlohmann_json::nlohmann_json)
target_compile_options(mirena_control PRIVATE -O3 -Wall -Wextra -Wpedantic)
ament_target_dependencies(mirena_control
  rclcpp rclcpp_lifecycle geometry_msgs visualization_msgs tf2 tf2_ros mirena_common)

add_executable(mirena_control_inspection src/inspection_node.cpp)
target_compile_options(mirena_control_inspection PRIVATE -O2 -Wall -Wextra -Wpedantic)
ament_target_dependencies(mirena_control_inspection rclcpp mirena_common)

install(TARGETS mirena_control mirena_control_inspection
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
