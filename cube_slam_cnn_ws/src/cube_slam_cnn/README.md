# Description #
The main implementation for cuboid SLAM. It uses g2o to formulate object SLAM optimization.

**Authors:** Shichao Yang (shichaoy@andrew.cmu.edu)
https://github.com/shichaoy/cube_slam

# modification by benchun 20201105 #
add other score function: 3d IoU and CNN 
default function is to use 3d IoU with ground truth
cnn network parameter: https://drive.google.com/file/d/18EXDn3TWbrljfh4Ln5w0UUaj3wcE-Idx/view?usp=sharing

## Installation

### Prerequisites
This code contains several ros packages. We test it in **ROS indigo/kinetic, Ubuntu 14.04/16.04, Opencv 2/3**. Create or use existing a ros workspace.
```bash
mkdir -p ~/cubeslamcnn_ws/src
cd ~/cubeslamcnn_ws/src
catkin_init_workspace
git clone git@github.com:path/cube_slam_cnn.git //(change the links)
cd cube_slam_cnn
```
### Compile dependency g2o
```bash
sh install_dependenices.sh
```
### Compile
```bash
cd ~/cubeslamcnn_ws
catkin_make -j4
```

## Running #
```bash
source devel/setup.bash
roslaunch object_slam object_slam_example.launch
```
You will see results in Rviz. Default rviz file is for ros indigo. A kinetic version is also provided.

To run other methods, modify the flag object_slam_example.launch
