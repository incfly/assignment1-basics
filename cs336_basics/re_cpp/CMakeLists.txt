cmake_minimum_required(VERSION 3.18)
project(re_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(re2 CONFIG REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(_re_cpp MODULE re_cpp.cpp)
target_include_directories(_re_cpp PRIVATE "${PYTHON_INCLUDE_DIR}")
target_link_libraries(_re_cpp PRIVATE re2::re2)
if(APPLE)
  target_link_options(_re_cpp PRIVATE "-undefined" "dynamic_lookup")
endif()

add_executable(capture_perf capture_perf.cpp)
target_link_libraries(capture_perf PRIVATE re2::re2)

set_target_properties(
  _re_cpp
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX ""
  SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
)

set_target_properties(
  capture_perf
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
