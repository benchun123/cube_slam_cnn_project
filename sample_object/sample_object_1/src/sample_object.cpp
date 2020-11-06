// std c
#include <math.h>
#include <stdio.h>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <sample_object/sample_object.h>

#include <deque>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

using namespace Eigen;
using namespace std;


void cuboid::print_cuboid()
{
    std::cout << "printing cuboids info...." << std::endl;
    std::cout << "pos   " << pos.transpose() << std::endl;
    std::cout << "scale   " << scale.transpose() << std::endl;
    std::cout << "rotY   " << rotY << std::endl;
    std::cout << "box_config_type   " << box_config_type.transpose() << std::endl;
    std::cout << "box_corners_2d \n"
              << box_corners_2d << std::endl;
    std::cout << "box_corners_3d_world \n"
              << box_corners_3d_world << std::endl;
}

// make sure column size is given. no checks here. row will be adjusted automatically. if more cols given, will be zero.
template <class T>
bool read_all_number_txt(const std::string txt_file_name, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &read_number_mat)
{
    if (!std::ifstream(txt_file_name))
    {
        std::cout << "ERROR!!! Cannot read txt file " << txt_file_name << std::endl;
        return false;
    }
    std::ifstream filetxt(txt_file_name.c_str());
    int row_counter = 0;
    std::string line;
    if (read_number_mat.rows() == 0)
        read_number_mat.resize(100, 10);

    while (getline(filetxt, line))
    {
        T t;
        if (!line.empty())
        {
            std::stringstream ss(line);
            int colu = 0;
            while (ss >> t)
            {
                read_number_mat(row_counter, colu) = t;
                colu++;
            }
            row_counter++;
            if (row_counter >= read_number_mat.rows()) // if matrix row is not enough, make more space.
                read_number_mat.conservativeResize(read_number_mat.rows() * 2, read_number_mat.cols());
        }
    }
    filetxt.close();

    read_number_mat.conservativeResize(row_counter, read_number_mat.cols()); // cut into actual rows

    return true;
}
template bool read_all_number_txt(const std::string, MatrixXd &);
template bool read_all_number_txt(const std::string, MatrixXi &);


template <class T>
Eigen::Quaternion<T> zyx_euler_to_quat(const T &roll, const T &pitch, const T &yaw)
{
    T sy = sin(yaw * 0.5);
    T cy = cos(yaw * 0.5);
    T sp = sin(pitch * 0.5);
    T cp = cos(pitch * 0.5);
    T sr = sin(roll * 0.5);
    T cr = cos(roll * 0.5);
    T w = cr * cp * cy + sr * sp * sy;
    T x = sr * cp * cy - cr * sp * sy;
    T y = cr * sp * cy + sr * cp * sy;
    T z = cr * cp * sy - sr * sp * cy;
    return Eigen::Quaternion<T>(w, x, y, z);
}
template Eigen::Quaterniond zyx_euler_to_quat<double>(const double &, const double &, const double &);
template Eigen::Quaternionf zyx_euler_to_quat<float>(const float &, const float &, const float &);

template <class T>
void quat_to_euler_zyx(const Eigen::Quaternion<T> &q, T &roll, T &pitch, T &yaw)
{
    T qw = q.w();
    T qx = q.x();
    T qy = q.y();
    T qz = q.z();

    roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
    pitch = asin(2 * (qw * qy - qz * qx));
    yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}
template void quat_to_euler_zyx<double>(const Eigen::Quaterniond &, double &, double &, double &);
template void quat_to_euler_zyx<float>(const Eigen::Quaternionf &, float &, float &, float &);

template <class T>
void rot_to_euler_zyx(const Eigen::Matrix<T, 3, 3> &R, T &roll, T &pitch, T &yaw)
{
    pitch = asin(-R(2, 0));

    if (abs(pitch - M_PI / 2) < 1.0e-3)
    {
        roll = 0.0;
        yaw = atan2(R(1, 2) - R(0, 1), R(0, 2) + R(1, 1)) + roll;
    }
    else if (abs(pitch + M_PI / 2) < 1.0e-3)
    {
        roll = 0.0;
        yaw = atan2(R(1, 2) - R(0, 1), R(0, 2) + R(1, 1)) - roll;
    }
    else
    {
        roll = atan2(R(2, 1), R(2, 2));
        yaw = atan2(R(1, 0), R(0, 0));
    }
}
template void rot_to_euler_zyx<double>(const Matrix3d &, double &, double &, double &);
template void rot_to_euler_zyx<float>(const Matrix3f &, float &, float &, float &);

template <class T>
Eigen::Matrix<T, 3, 3> euler_zyx_to_rot(const T &roll, const T &pitch, const T &yaw)
{
    T cp = cos(pitch);
    T sp = sin(pitch);
    T sr = sin(roll);
    T cr = cos(roll);
    T sy = sin(yaw);
    T cy = cos(yaw);

    Eigen::Matrix<T, 3, 3> R;
    R << cp * cy, (sr * sp * cy) - (cr * sy), (cr * sp * cy) + (sr * sy),
        cp * sy, (sr * sp * sy) + (cr * cy), (cr * sp * sy) - (sr * cy),
        -sp, sr * cp, cr * cp;
    return R;
}
template Matrix3d euler_zyx_to_rot<double>(const double &, const double &, const double &);
template Matrix3f euler_zyx_to_rot<float>(const float &, const float &, const float &);


template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> real_to_homo_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_homo_out;
    int raw_rows = pts_in.rows();
    int raw_cols = pts_in.cols();

    pts_homo_out.resize(raw_rows + 1, raw_cols);
    pts_homo_out << pts_in,
        Matrix<T, 1, Dynamic>::Ones(raw_cols);
    return pts_homo_out;
}
template MatrixXd real_to_homo_coord<double>(const MatrixXd &);
template MatrixXf real_to_homo_coord<float>(const MatrixXf &);

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> homo_to_real_coord(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &pts_homo_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pts_out(pts_homo_in.rows() - 1, pts_homo_in.cols());
    for (int i = 0; i < pts_homo_in.rows() - 1; i++)
        pts_out.row(i) = pts_homo_in.row(i).array() / pts_homo_in.bottomRows(1).array(); //replicate needs actual number, cannot be M or N

    return pts_out;
}
template MatrixXd homo_to_real_coord<double>(const MatrixXd &);
template MatrixXf homo_to_real_coord<float>(const MatrixXf &);

template <class T> // though vector can be casted into matrix, to make output clear to be vector, it is better to define a new function.
Eigen::Matrix<T, Eigen::Dynamic, 1> homo_to_real_coord_vec(const Eigen::Matrix<T, Eigen::Dynamic, 1> &pts_homo_in)
{
    Eigen::Matrix<T, Eigen::Dynamic, 1> pt_out;
    if (pts_homo_in.rows() == 4)
        pt_out = pts_homo_in.head(3) / pts_homo_in(3);
    else if (pts_homo_in.rows() == 3)
        pt_out = pts_homo_in.head(2) / pts_homo_in(2);
    return pt_out;
}
template VectorXd homo_to_real_coord_vec<double>(const VectorXd &);
template VectorXf homo_to_real_coord_vec<float>(const VectorXf &);

Matrix4d similarityTransformation(const cuboid &cube_obj)
{
    Matrix3d rot;
    rot << cos(cube_obj.rotY), -sin(cube_obj.rotY), 0,
        sin(cube_obj.rotY), cos(cube_obj.rotY), 0,
        0, 0, 1;
    Matrix3d scale_mat = cube_obj.scale.asDiagonal();

    Matrix4d res = Matrix4d::Identity();
    res.topLeftCorner<3, 3>() = rot * scale_mat;
    res.col(3).head(3) = cube_obj.pos;
    return res;
}

Eigen::Matrix3Xd compute3D_BoxCorner(const cuboid &cube_obj)
{
    MatrixXd corners_body;
    corners_body.resize(3, 8);
    corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
        1, -1, -1, 1, 1, -1, -1, 1,
        -1, -1, -1, -1, 1, 1, 1, 1;
    MatrixXd corners_world = homo_to_real_coord<double>(similarityTransformation(cube_obj) * real_to_homo_coord<double>(corners_body));
    return corners_world;
}

Eigen::MatrixXi compute2D_BoxCorner(const cuboid &cube_obj, const Eigen::Matrix<double, 3, 4>& projectMatrix)
{
    Eigen::MatrixXi corners_2d_return(2, 8);    // same type with cube.box_corners_2d
    Eigen::Matrix<double, 3, 8> corners_2d;
    Eigen::Matrix<double, 4, 8> corners_3d;
    corners_3d.block(0,0,3,8) = cube_obj.box_corners_3d_world;
    for (size_t i = 0; i < corners_3d.cols(); i++)
        corners_3d(3,i) = 1.0;

    corners_2d = projectMatrix  * corners_3d ;
    for (size_t i = 0; i < corners_2d.cols(); i++)
    {
        corners_2d(0,i) = corners_2d(0,i) /corners_2d(2,i);
        corners_2d(1,i) = corners_2d(1,i) /corners_2d(2,i);
        corners_2d(2,i) = corners_2d(2,i) /corners_2d(2,i);
    }
    corners_2d_return = corners_2d.topRows(2).cast <int> ();
    return corners_2d_return;
}

 void plot_image_with_cuboid_new(cv::Mat &plot_img, const cuboid *cube_obj)
 {
    Eigen::MatrixXi edge_order(2, 12); // normally, the corners are in order and edges are in order
    edge_order << 1, 2, 3, 4, 1, 3, 5, 7, 1, 2, 5, 6,  // z axis: 15, 26, ... x axis: 12, 34, ... y axis: 14, 23, ...
                  5, 6, 7, 8, 2, 4, 6, 8, 4, 3, 8, 7;
    cv::Scalar color = cv::Scalar(0, 255, 0); // green
    int thickness = 2; 
    int lineType = CV_AA; // 4, 8, or CV_AA
    Eigen::Matrix2Xi box_corners_2d = cube_obj->box_corners_2d;
    for (int edge_id = 0; edge_id < edge_order.cols(); edge_id++)
    {
        cv::Point pt0 = cv::Point (box_corners_2d(0, edge_order(0, edge_id)-1), box_corners_2d(1, edge_order(0, edge_id)-1));
        cv::Point pt1 = cv::Point (box_corners_2d(0, edge_order(1, edge_id)-1), box_corners_2d(1, edge_order(1, edge_id)-1));
        cv::line(plot_img, pt0, pt1, color, thickness, lineType, 0);
    }
    // for (size_t i = 0; i < cube_obj->box_corners_2d.cols(); i++)
    // {
    //     cv::circle(plot_img, cv::Point(cube_obj->box_corners_2d(0,i),cube_obj->box_corners_2d(1,i)),
    //                 i,cv::Scalar(0, 255, 0),1,8, 0);
    // }
 }

//rays is 3*n, each column is a ray staring from origin  plane is (4，1） parameters, compute intersection  output is 3*n
void ray_plane_interact(const MatrixXd &rays, const Eigen::Vector4d &plane, MatrixXd &intersections)
{
    VectorXd frac = -plane[3] / (plane.head(3).transpose() * rays).array(); //n*1
    intersections = frac.transpose().replicate<3, 1>().array() * rays.array();
}

void plane_hits_3d(const Matrix4d &transToWolrd, const Matrix3d &invK, const Vector4d &plane_sensor, MatrixXd pixels, Matrix3Xd &pts_3d_world)
// compute ray intersection with plane in 3D.
// transToworld: 4*4 camera pose.   invK: inverse of calibration.   plane: 1*4  plane equation in sensor frame.
// pixels  2*n; each column is a pt [x;y] x is horizontal,y is vertical   outputs: pts3d 3*n in world frame
{
    pixels.conservativeResize(3, NoChange);
    pixels.row(2) = VectorXd::Ones(pixels.cols());
    MatrixXd pts_ray = invK * pixels; //each column is a 3D world coordinate  3*n
    MatrixXd pts_3d_sensor;
    ray_plane_interact(pts_ray, plane_sensor, pts_3d_sensor);
    pts_3d_world = homo_to_real_coord<double>(transToWolrd * real_to_homo_coord<double>(pts_3d_sensor)); //
}

Vector4d get_wall_plane_equation(const Vector3d &gnd_seg_pt1, const Vector3d &gnd_seg_pt2)
// 1*6 a line segment in 3D. [x1 y1 z1  x2 y2 z2]  z1=z2=0  or  two 1*3
{

    Vector3d partwall_normal_world = (gnd_seg_pt1 - gnd_seg_pt2).cross(Vector3d(0, 0, 1)); // [0,0,1] is world ground plane
    partwall_normal_world.array() /= partwall_normal_world.norm();
    double dist = -partwall_normal_world.transpose() * gnd_seg_pt1;
    Vector4d plane_equation;
    plane_equation << partwall_normal_world,
        dist; // wall plane in world frame
    if (dist < 0)
        plane_equation = -plane_equation; // make all the normal pointing inside the room. neamly, pointing to the camera
    return plane_equation;
}

 // box_corners_2d_float is 2*8    change to my object struct from 2D box corners.
void change_2d_corner_to_3d_object(const MatrixXd &box_corners_2d_float, const Vector3d &configs, const Vector4d &ground_plane_sensor,
                                   const Matrix4d &transToWolrd, const Matrix3d &invK, Eigen::Matrix<double, 3, 4> &projectionMatrix,
                                   cuboid &sample_obj)
{
    Matrix3Xd obj_gnd_pt_world_3d;
    plane_hits_3d(transToWolrd, invK, ground_plane_sensor, box_corners_2d_float.rightCols(4), obj_gnd_pt_world_3d); //% 3*n each column is a 3D point  floating point

    double length_half = (obj_gnd_pt_world_3d.col(0) - obj_gnd_pt_world_3d.col(3)).norm() / 2; // along object x direction   corner 5-8
    double width_half = (obj_gnd_pt_world_3d.col(0) - obj_gnd_pt_world_3d.col(1)).norm() / 2;  // along object y direction   corner 5-6

    Vector4d partwall_plane_world = get_wall_plane_equation(obj_gnd_pt_world_3d.col(0), obj_gnd_pt_world_3d.col(1)); //% to compute height, need to unproject-hit-planes formed by 5-6 corner
    Vector4d partwall_plane_sensor = transToWolrd.transpose() * partwall_plane_world;                                // wall plane in sensor frame

    Matrix3Xd obj_top_pt_world_3d;
    plane_hits_3d(transToWolrd, invK, partwall_plane_sensor, box_corners_2d_float.col(1), obj_top_pt_world_3d); // should match obj_gnd_pt_world_3d  % compute corner 2
    double height_half = obj_top_pt_world_3d(2, 0) / 2;

    double mean_obj_x = obj_gnd_pt_world_3d.row(0).mean();
    double mean_obj_y = obj_gnd_pt_world_3d.row(1).mean();

    double vp_1_position = configs(1);
    double yaw_esti = configs(2);
    sample_obj.pos = Vector3d(mean_obj_x, mean_obj_y, height_half);
    sample_obj.rotY = yaw_esti;
    sample_obj.scale = Vector3d(length_half, width_half, height_half);
    sample_obj.box_config_type = configs.head<2>();
    VectorXd cuboid_to_raw_boxstructIds(8);
    if (vp_1_position == 1) // vp1 on left, for all configurations
        cuboid_to_raw_boxstructIds << 6, 5, 8, 7, 2, 3, 4, 1;
    if (vp_1_position == 2) // vp1 on right, for all configurations
        cuboid_to_raw_boxstructIds << 5, 6, 7, 8, 3, 2, 1, 4;

    Matrix2Xi box_corners_2d_int = box_corners_2d_float.cast<int>();
    sample_obj.box_corners_2d.resize(2, 8);
    for (int i = 0; i < 8; i++)
        sample_obj.box_corners_2d.col(i) = box_corners_2d_int.col(cuboid_to_raw_boxstructIds(i) - 1); // minius one to match index

    sample_obj.box_corners_3d_world = compute3D_BoxCorner(sample_obj);
}



double box3d_iou(const cuboid *sample, const cuboid *ground)
{
    // get 2d area in the top
    // append outer and inners() https://www.boost.org/doc/libs/1_65_1/libs/geometry/doc/html/geometry/reference/models/model_polygon.html
    typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> point_t;
    typedef boost::geometry::model::polygon<point_t> polygon_t;
    polygon_t sam_poly, gt_poly;
    Eigen::Matrix<double, 3, 8> sam_corner_3d = sample->box_corners_3d_world;
    Eigen::Matrix<double, 3, 8> gt_corner_3d = ground->box_corners_3d_world;
    for (size_t i = 0; i < 4; i++)
    {
      point_t sam_top_points(sam_corner_3d(0,i),sam_corner_3d(1,i));
      point_t gt_top_points(gt_corner_3d(0,i),gt_corner_3d(1,i));
      boost::geometry::append(sam_poly.outer(), sam_top_points);
      boost::geometry::append(gt_poly.outer(), gt_top_points);
      if (i == 3) // add start point to make a closed form
      {
        boost::geometry::append(sam_poly.outer(), point_t(sam_corner_3d(0,0),sam_corner_3d(1,0)));
        boost::geometry::append(gt_poly.outer(), point_t(gt_corner_3d(0,0),gt_corner_3d(1,0)));    
      }
    }
    std::vector<polygon_t> inter_poly;
    boost::geometry::intersection(sam_poly, gt_poly, inter_poly); 
    double inter_area = inter_poly.empty() ? 0 : boost::geometry::area(inter_poly.front());
    double union_area = boost::geometry::area(sam_poly) + boost::geometry::area(gt_poly) - inter_area;// boost::geometry::union_(poly1, poly2, un);
    double iou_2d = inter_area / union_area;
    // std::cout << "iou2d: " << iou_2d << std::endl;

    double h_up = min(sam_corner_3d(2,4),gt_corner_3d(2,4));
    double h_down = max(sam_corner_3d(2,0),gt_corner_3d(2,0));
    double inter_vol = inter_area * max(0.0, h_up - h_down);
    double sam_vol = sample->scale(0)*2 * sample->scale(1)*2 * sample->scale(2)*2;
    double gt_vol = ground->scale(0)*2 * ground->scale(1)*2 * ground->scale(2)*2;
    double iou_3d = inter_vol / (sam_vol + gt_vol - inter_vol);
    // std::cout << "iou3d: " << iou_3d << std::endl;
    return iou_3d;
}