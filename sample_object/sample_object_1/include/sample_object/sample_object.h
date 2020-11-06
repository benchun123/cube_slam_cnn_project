#pragma once

// std c
#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>

class cuboid // matlab cuboid struct. cuboid on ground. only has yaw, no obj roll/pitch
{
    public:
      Eigen::Vector3d pos;
      Eigen::Vector3d scale;
      double rotY;

      Eigen::Vector2d box_config_type;       // configurations, vp1 left/right
      Eigen::Matrix2Xi box_corners_2d;       // 2*8
      Eigen::Matrix3Xd box_corners_3d_world; // 3*8

      Eigen::Vector4d rect_detect_2d; //% 2D bounding box (might be expanded by me)
      double edge_distance_error;
      double edge_angle_error;
      double normalized_error; // normalized distance+angle
      double skew_ratio;
      double down_expand_height;
      double camera_roll_delta;
      double camera_pitch_delta;

      void print_cuboid(); // print pose information
};
typedef std::vector<cuboid *> ObjectSet; // for each 2D box, the set of generated 3D cuboids

struct cam_pose_infos
{
      Eigen::Matrix4d transToWolrd;
      Eigen::Matrix3d Kalib;

      Eigen::Matrix3d rotationToWorld;
      Eigen::Vector3d euler_angle;
      Eigen::Matrix3d invR;
      Eigen::Matrix3d invK;
      Eigen::Matrix<double, 3, 4> projectionMatrix;
      Eigen::Matrix3d KinvR; // K*invR
      double camera_yaw;
};


template <class T>
Eigen::Quaternion<T> zyx_euler_to_quat(const T &roll, const T &pitch, const T &yaw);

template <class T>
void quat_to_euler_zyx(const Eigen::Quaternion<T> &q, T &roll, T &pitch, T &yaw);

template <class T>
void rot_to_euler_zyx(const Eigen::Matrix<T, 3, 3> &R, T &roll, T &pitch, T &yaw);

template <class T>
Eigen::Matrix<T, 3, 3> euler_zyx_to_rot(const T &roll, const T &pitch, const T &yaw);

// input is 3*n (or 2*n)  output is 4*n (or 3*n)
template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in);
template <class T>
void real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_out);
template <class T> // though vector can be casted into matrix, to make output clear to be vector, it is better to define a new function.
Eigen::Matrix<T, Eigen::Dynamic, 1> real_to_homo_coord_vec(const Eigen::Matrix<T, Eigen::Dynamic, 1> &pts_in);

// input is 3*n (or 4*n)  output is 2*n(or 3*n)
template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in);
template <class T>
void homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_out);

template <class T> // though vector can be casted into matrix, to make output clear to be vector, it is better to define a new function.
Eigen::Matrix<T, Eigen::Dynamic, 1> homo_to_real_coord_vec(const Eigen::Matrix<T, Eigen::Dynamic, 1> &pts_homo_in);

// make sure column size is given. not check here. row will be adjusted automatically. if more cols given, will be zero.
template <class T>
bool read_all_number_txt(const std::string txt_file_name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &read_number_mat);

Eigen::Matrix4d similarityTransformation(const cuboid &cube_obj);
Eigen::Matrix3Xd compute3D_BoxCorner(const cuboid &cube_obj);
Eigen::MatrixXi compute2D_BoxCorner(const cuboid &cube_obj, const Eigen::Matrix<double, 3, 4>& projectMatrix);
void plot_image_with_cuboid_new(cv::Mat &plot_img, const cuboid *cube_obj);

void ray_plane_interact(const Eigen::MatrixXd &rays, const Eigen::Vector4d &plane, Eigen::MatrixXd &intersections);
void plane_hits_3d(const Eigen::Matrix4d &transToWolrd, const Eigen::Matrix3d &invK, const Eigen::Vector4d &plane_sensor, Eigen::MatrixXd pixels, Eigen::Matrix3Xd &pts_3d_world);
Eigen::Vector4d get_wall_plane_equation(const Eigen::Vector3d &gnd_seg_pt1, const Eigen::Vector3d &gnd_seg_pt2);
void change_2d_corner_to_3d_object(const Eigen::MatrixXd &box_corners_2d_float, const Eigen::Vector3d &configs, const Eigen::Vector4d &ground_plane_sensor,
                                   const Eigen::Matrix4d &transToWolrd, const Eigen::Matrix3d &invK, Eigen::Matrix<double, 3, 4> &projectionMatrix,
                                   cuboid &sample_obj);

double box3d_iou(const cuboid *sample, const cuboid *ground);



