# Description #
This package contains to single image cuboid detection in C++. Given 2D object detection, it generates many 3D cuboid proposal and selects the best one. It matches a [matlab implementation](https://github.com/shichaoy/matlab_cuboid_detect). Due to different canny edge and distancen transform, the final output might be slightly differently. For understanding and debuging purpose, it is suggested to use matlab implementation.



**Authors:** Shichao Yang (shichaoy@andrew.cmu.edu)

# How to run.
1. catkin_make.
2. `rosrun detect_3d_cuboid detect_3d_cuboid_node`

See the main file ```src/main.cpp``` for more details.


# modification by benchun 20201105 #
change catkin_make (ros) file to cmake file (comment out all ros sentence) 
# How to run.
1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./detect_3d_cuboid .../data

PS: 
when you use `make` command and meet problem with `error: ‘iota’ was not declared in this scope`, it is not about the code problem but the compiler bug, I don't how to fix it, but to comment out line 3 ('set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )') in CMakeLists.txt, make again, then, you can also uncomment this sentence. the error disappear...

# Task 1#
1st week: result: draw all the proposals with different image and parameters, mainly in 2d
q1: how geometry method samples the bounding box (object yaw, top corner, ...)
q2: how to select best bounding box
q3: which parameters influence the sample process (object yaw, top corner, …)
q4: how to change 2d corners to 3d corners

hints: 
1) replace Kalib, transToWolrd, 2d bbox, image and edge_detection
2) the dataset is fr3/cabinet from website: https://vision.in.tum.de/data/datasets/rgbd-dataset/download, from the website you can find useful information such as the calibration, and the formation of Ground-truth trajectories (truth_cam_poses.txt)
3) to draw the all the proposals, you should consider the parameters: max_cuboid_num,  whether_sample_cam_roll_pitch
