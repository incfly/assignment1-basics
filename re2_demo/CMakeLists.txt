cmake_minimum_required(VERSION 3.18)
project(re2_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(re2 CONFIG REQUIRED)

Python3_add_library(_re2demo MODULE re2_demo.cpp)
target_link_libraries(_re2demo PRIVATE re2::re2)

set_target_properties(
  _re2demo
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX ""
)
