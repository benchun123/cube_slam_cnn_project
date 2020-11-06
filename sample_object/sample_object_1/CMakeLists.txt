cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
project(sample_object)

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

SET(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
# C++11

## Declare a C++ library
add_library(${PROJECT_NAME}  SHARED src/sample_object.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(detect_3d_cuboid ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
add_executable(sample_object_node src/main_sample_object.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(detect_3d_cuboid_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
#target_link_libraries(detect_3d_cuboid_node_debug  ${PROJECT_NAME}
#   ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES}
#)

target_link_libraries(sample_object_node  
   ${PROJECT_NAME}
   ${OpenCV_LIBRARIES} 
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_detect_3d_cuboid.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
