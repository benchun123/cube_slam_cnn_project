cmake_minimum_required(VERSION 2.8)
project(detect_3d_cuboid)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( CMAKE_BUILD_TYPE Release)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
# find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( "/usr/include/eigen3" )

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
)

## Declare a C++ library
add_library(${PROJECT_NAME}  SHARED
src/box_proposal_detail.cpp src/object_3d_util.cpp src/matrix_utils.cpp  
)
target_link_libraries(${PROJECT_NAME}   ${OpenCV_LIBRARIES} )

## Declare a C++ executable
add_executable(detect_3d_cuboid_node src/main.cpp)

target_link_libraries(detect_3d_cuboid_node  
   ${PROJECT_NAME}   ${OpenCV_LIBRARIES} 
)

# add_executable(draw_3d_cuboid_node src/main_draw_cube_offline.cpp)

#target_link_libraries(draw_3d_cuboid_node  ${PROJECT_NAME}
#   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES}
# )

