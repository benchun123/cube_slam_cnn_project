cmake_minimum_required(VERSION 2.8.3)
project(sample_object)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( CMAKE_BUILD_TYPE Release)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
# find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( "/usr/include/eigen3" )

# tictoc_profiler: timer
set( TICTOC_DIRS ${PROJECT_SOURCE_DIR}/third_party/tictoc_profiler/include)
set( TICTOC_LIBS ${PROJECT_SOURCE_DIR}/third_party/tictoc_profiler/libtictoc_profiler.so)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}  
  ${TICTOC_DIRS}
)

SET(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
# C++11

## Declare a C++ library
add_library(${PROJECT_NAME}  SHARED
src/box_proposal_detail.cpp src/object_3d_util.cpp src/matrix_utils.cpp  
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}  ${TICTOC_LIBS} ${Boost_SYSTEM_LIBRARY}
)

## Declare a C++ executable
add_executable(sample_random_cuboid_node src/main_sample_random.cpp)
add_executable(sample_ground_truth_node src/main_sample_ground_truth.cpp)


target_link_libraries(sample_random_cuboid_node  ${PROJECT_NAME}
   ${OpenCV_LIBRARIES}  ${TICTOC_LIBS} ${Boost_SYSTEM_LIBRARY}
)

target_link_libraries(sample_ground_truth_node  ${PROJECT_NAME}
   ${OpenCV_LIBRARIES}  ${TICTOC_LIBS} ${Boost_SYSTEM_LIBRARY}
)


