# Description #
This package contains to single image cuboid detection in C++. Given 2D object detection, it generates many 3D cuboid proposal and selects the best one. It matches a [matlab implementation](https://github.com/shichaoy/matlab_cuboid_detect). Due to different canny edge and distancen transform, the final output might be slightly differently. For understanding and debuging purpose, it is suggested to use matlab implementation.



**Authors:** Shichao Yang (shichaoy@andrew.cmu.edu)

# How to run.
1. catkin_make.
2. `rosrun detect_3d_cuboid detect_3d_cuboid_node`

See the main file ```src/main.cpp``` for more details.

# modification by benchun 20201105 #
plot all the samples without score function with line, add other score functions with ground truth, including MSE, 3d IoU. 
PS: search "comment out score or sort function benchun 20200914" in box_proposal.detail.cpp, object_3d_util.cpp
PS: added timer in third party folder, make the library in tictoc_profiler folder. 

# How to run.
1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./score_3d_cuboid_node ../data/

# Task 2 #
3rd week: result: give the ground truth, select best offline bounding box and evaluate in slam
q2: how to select best bounding box with ground truth (mse, 3d iou)
q3: how to improve sample process: modify 2d bounding box, sample camera yaw,


# Task 4 #
apply pytorch score network in C++. 
search "added libtorch if needed 20201105 benchun" in CMakeList.txt, main, ...

