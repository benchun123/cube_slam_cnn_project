cmake_minimum_required(VERSION 2.8)
project(score_3d_cuboid)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( CMAKE_BUILD_TYPE Release)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
# find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( "/usr/include/eigen3" )

# added libtorch if needed 20201105 benchun 
# set(CMAKE_PREFIX_PATH /home/benchun/Software/libtorch)
# find_package(Torch REQUIRED)
# message(${TORCH_LIBRARIES})
# message(${TORCH_INCLUDE_DIRS})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DBOOST_MATH_DISABLE_FLOAT128")

# added libtorch if needed 20201105 benchun
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
  # ${TORCH_INCLUDE_DIRS} # added libtorch if needed 20201105 benchun
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
    # ${TORCH_LIBRARIES} # added libtorch if needed 20201105 benchun
    ${OpenCV_LIBRARIES} ${TICTOC_LIBS} ${Boost_SYSTEM_LIBRARY}
)


## Declare a C++ executable
add_executable(detect_3d_cuboid_node src/main.cpp)
add_executable(score_3d_cuboid_node src/main_score_cuboid.cpp)


target_link_libraries(detect_3d_cuboid_node  ${PROJECT_NAME}
    ${OpenCV_LIBRARIES} 
)
target_link_libraries(score_3d_cuboid_node  ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}  ${TORCH_LIBRARIES}  ${Boost_SYSTEM_LIBRARY}
    # ${TICTOC_LIBS}  # added libtorch if needed 20201105 benchun
)

# added libtorch if needed 20201105 benchun
# set_property(TARGET score_3d_cuboid_node PROPERTY CXX_STANDARD 14)

