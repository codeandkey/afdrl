cmake_minimum_required (VERSION 3.0)

project (AFDRL)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(ale REQUIRED)
find_package(Torch REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)

set (CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_STANDARD 17)

add_executable(afdrl
  afdrl/afdrl.cpp
  afdrl/env.cpp
  afdrl/schedule.cpp
  afdrl/train.cpp
  afdrl/test.cpp
  afdrl/agent.cpp
)

target_link_libraries(afdrl
  ${TORCH_LIBRARIES}
  ale::ale-lib
  MPI::MPI_CXX
  ${OpenCV_LIBS}
)

target_precompile_headers(afdrl PUBLIC
  afdrl/torch_pch.h
)

# what are the opencv include dirs?
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

target_include_directories(afdrl PUBLIC
  ${TORCH_INCLUDE_DIRS}
  ${ALE_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
