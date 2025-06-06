cmake_minimum_required(VERSION 3.8)
project(cfar)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Boost REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Python bindings
pybind11_add_module(pycfar
    src/cfar.cpp
    src/cfar_bindings.cpp
    src/utils/ndarray_converter.cpp
)

target_link_libraries(pycfar 
  PRIVATE 
    ${OpenCV_LIBS}
    ${ARMADILLO_LIBRARIES}
    ${Boost_LIBRARIES}
)

target_include_directories(pycfar
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

set_target_properties(pycfar PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    # INSTALL_RPATH "${OpenCV_LIBRARY_DIRS}"
)

# CFAR library
# This library can be used independently of the Python bindings
# and can be linked to others C++ projects.
add_library(cfar src/cfar.cpp)
target_include_directories(cfar 
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    # armadillo
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(cfar 
  PRIVATE 
    armadillo  
    ${Boost_LIBRARIES} 
    ${BLAS_LIBRARIES} 
    ${LAPACK_LIBRARIES}
)

install(
  TARGETS cfar
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS pycfar
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)