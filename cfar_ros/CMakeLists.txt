cmake_minimum_required(VERSION 3.8)
project(cfar_ros)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set RPATH settings
# set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# Set output directories
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CfarInfo.msg"
  DEPENDENCIES std_msgs
)

add_subdirectory(../cfar ${CMAKE_CURRENT_BINARY_DIR}/cfar_build)

add_executable(cfar_node src/cfar_node.cpp)
ament_target_dependencies(cfar_node 
  rclcpp 
  std_msgs 
  sensor_msgs 
  cv_bridge
  image_transport
)

rosidl_get_typesupport_target(
  cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

target_link_libraries(cfar_node
  ${cpp_typesupport_target}
  ${OpenCV_LIBRARIES}
  cfar
)

# (Optional) Standalone test executable – not installed
# add_executable(cfar src/main.cpp)
# target_link_libraries(cfar
#   ${OpenCV_LIBRARIES}
#   cfar_lib
# )

install(
  TARGETS cfar_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY launch cfg
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
