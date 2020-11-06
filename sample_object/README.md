# Description #
This package contains to single image cuboid detection in C++. Given 2D object detection, it generates many 3D cuboid proposal and selects the best one. It matches a [matlab implementation](https://github.com/shichaoy/matlab_cuboid_detect). Due to different canny edge and distancen transform, the final output might be slightly differently. For understanding and debuging purpose, it is suggested to use matlab implementation.



**Authors:** Shichao Yang (shichaoy@andrew.cmu.edu)

# How to run.
1. catkin_make.
2. `rosrun detect_3d_cuboid detect_3d_cuboid_node`

See the main file ```src/main.cpp``` for more details.

# modification by benchun 20201105 #
sample cuboid and make dataset
we provide two methods to make dataset, random sampling and based on ground truth
random sampling is just follow how to sample proposals
grounth truth is just add some noise on xyz 

# How to run.
1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./sample_ground_truth_node ../data/
./sample_random_cuboid_node ../data/

hint: 
1) whenever you want to make the dataset, change the `dataset_folder` to where you want, and make `image_2` and `label_2` folder, then, open the flag to save image and save txt. 
2) in random sampling, we also add `whether_check_data_balance` flag to make a balance dataset. we divided (0,1) into 20 pieces, and each piece have maximum 2 or 3 image. 
3) 