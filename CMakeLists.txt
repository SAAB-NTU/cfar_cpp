cmake_minimum_required(VERSION 3.8)
project(cfar_cpp)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set RPATH settings
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# Set output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(ament_cmake_python REQUIRED)


find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pycfar
    src/cfar.cpp
    src/cfar_bindings.cpp
    src/utils/ndarray_converter.cpp
)

target_link_libraries(pycfar 
  PRIVATE 
    ${OpenCV_LIBS}
    # ${Armadillo_LIBRARIES}
    armadillo
    ${Boost_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

target_include_directories(pycfar
  PRIVATE
    ${pybind11_INCLUDE_DIRS}
)

set_target_properties(pycfar PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${OpenCV_LIBRARY_DIRS}"
)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CfarInfo.msg"
  DEPENDENCIES std_msgs
)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Create the library
add_library(cfar_lib src/cfar.cpp)
target_include_directories(cfar_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include # For headers exposed by cfar_lib
    ${Eigen3_INCLUDE_DIRS}              # Eigen is header-only, so its includes are needed during compilation of cfar.cpp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src     # For internal source files
    ${ARMADILLO_INCLUDE_DIRS}           # Armadillo includes for cfar.cpp
)

target_link_libraries(cfar_lib PRIVATE ${ARMADILLO_LIBRARIES})

# Create executables
add_executable(cfar src/main.cpp)
add_executable(cfar_node src/cfar_node.cpp src/cfar.cpp)

ament_target_dependencies(cfar_node 
  rclcpp 
  std_msgs 
  sensor_msgs 
  cv_bridge
  image_transport
)

# Get the typesupport target
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(cfar
  ${OpenCV_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  cfar_lib
)

target_link_libraries(cfar_node
  ${cpp_typesupport_target}
  ${OpenCV_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  cfar_lib
)

# Install targets
install(
  TARGETS cfar_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS cfar cfar_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS pycfar
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY cfg
  DESTINATION share/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
