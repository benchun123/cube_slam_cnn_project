// std c
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

#include <deque>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

// ours
#include "detect_3d_cuboid/detect_3d_cuboid.h"
#include "detect_3d_cuboid/matrix_utils.h"
#include "detect_3d_cuboid/object_3d_util.h"
#include "tictoc_profiler/profiler.hpp"

namespace bg = boost::geometry;

using namespace std;
using namespace cv;
using namespace Eigen;

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
    Eigen::Matrix2Xi box_corners_2d = cube_obj->box_corners_2d;
    for (int edge_id = 0; edge_id < edge_order.cols(); edge_id++)
    {
        cv::Point pt0 = cv::Point (box_corners_2d(0, edge_order(0, edge_id)-1), box_corners_2d(1, edge_order(0, edge_id)-1));
        cv::Point pt1 = cv::Point (box_corners_2d(0, edge_order(1, edge_id)-1), box_corners_2d(1, edge_order(1, edge_id)-1));
        cv::line(plot_img, pt0, pt1, cv::Scalar(0, 255, 0), 2, CV_AA, 0);
    }
    // for (size_t i = 0; i < cube_obj->box_corners_2d.cols(); i++)
    // {
    //     cv::circle(plot_img, cv::Point(cube_obj->box_corners_2d(0,i),cube_obj->box_corners_2d(1,i)),
    //                 i,cv::Scalar(0, 255, 0),1,8, 0);
    // }
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
    // std::cout << "iou3d: " << iou_3d <<  std::endl;
    return iou_3d;
    // Eigen::Vector3d pos1 = sample->pos;
    // Eigen::Vector3d pos2 = ground->pos;
    // double pose_error = sqrt(pow(pos1(0)-pos2(0),2)+pow(pos1(1)-pos2(1),2)+pow(pos1(2)-pos2(2),2));
    // return iou_3d-5*pose_error;
}

int main(int argc, char **argv)
{

  bool whether_plot_final_images = true;
  bool whether_save_final_images = false;
  bool whether_save_cubuid_txt = false;
  bool whether_sample_images = false;

  string base_folder = argv[1];
  string imgs_folder = base_folder + "/raw_imgs/";
//   string save_folder = "/home/benchun/benchun/dataset/cubes/training_2/";
  string save_folder = base_folder;

  // std::string pred_objs_txt = base_folder + "0000_3d_cuboids.txt"; // data: x y z yaw length width height
  std::string pred_objs_txt = base_folder+"online_camera_pose/online_cubes.txt";  // data: x, y, z, pitch, toll, yaw, length, width, height
  std::string init_camera_pose = base_folder+"online_camera_pose/online_camera.txt"; // // time x y z qx qy qz qw
  std::string truth_camera_pose = base_folder+"online_camera_pose/truth_cam_poses.txt";  // data: x, y, z, qx, qy, qz, qw
  Eigen::MatrixXd pred_frame_objects(100,10);  // 100 is some large row number, each row in txt has 10 numbers
  Eigen::MatrixXd init_frame_poses(100,8);
  Eigen::MatrixXd truth_frame_poses(100,8);
  if (!read_all_number_txt(pred_objs_txt, pred_frame_objects))
	  return -1;
  if (!read_all_number_txt(init_camera_pose,init_frame_poses))
	return -1;
  if (!read_all_number_txt(truth_camera_pose,truth_frame_poses))
	  return -1;
  int total_frame_number = truth_frame_poses.rows();

  Eigen::Matrix3d Kalib;
  Kalib<< 535.4,  0,  320.1,   // for TUM cabinet data.
          0,  539.2, 247.6,
          0,      0,     1;

  total_frame_number = 1;
  for (int frame_index = 0; frame_index < total_frame_number; frame_index++)
  {
    // frame_index = 3;
    char frame_index_c[256];
    sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit
    cv::Mat rgb_img = cv::imread(imgs_folder + frame_index_c + "_rgb_raw.jpg", 1);
    std::cout << "frame_index: " << frame_index << std::endl;

    //read cleaned yolo 2d object detection
    Eigen::MatrixXd raw_2d_objs(10,5);  // 2d rect [x1 y1 width height], and prob
    raw_2d_objs.setZero();
    if (!read_all_number_txt(base_folder+"/filter_2d_obj_txts/"+frame_index_c+"_yolo2_0.15.txt", raw_2d_objs))
    return -1;
    raw_2d_objs.leftCols<2>().array() -=1;   // change matlab coordinate to c++, minus 1
    // std::cout << "raw_2d_objs" << raw_2d_objs << std::endl;
    if(!raw_2d_objs.isZero())
    {

    Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
    // Eigen::MatrixXd cam_pose_Twc = init_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
    Matrix<double,4,4> transToWolrd;
    transToWolrd.setIdentity();
    transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
    transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
    Eigen::Matrix<double, 3, 4> projectionMatrix = Kalib * transToWolrd.inverse().topRows<3>(); // project world coordinate to camera
    // std::cout << "transToWolrd: \n" << transToWolrd << std::endl;
    // std::cout << "projectionMatrix origin: \n" << projectionMatrix << std::endl;
    Eigen::Vector3d orientation;
    rot_to_euler_zyx<double>(transToWolrd.block(0,0,3,3), orientation(0), orientation(1), orientation(2));
    std::cout << "orientation: " << orientation.transpose() << std::endl;

    // todo use gaussian distribution to sample the data // really need to sample z axis?
    Eigen::MatrixXd sample_x_list, sample_y_list, sample_z_list, sample_yaw_list;
    sample_x_list.resize(1,5);
    sample_x_list << 0, -0.03, 0.03, -0.06, 0.06;
    sample_y_list = sample_x_list;
    sample_z_list = sample_y_list;
    sample_yaw_list = sample_x_list;

    if (whether_sample_images == false)
    {
      sample_x_list.resize(1,1);
      sample_x_list(0,0)=0;
      sample_y_list = sample_x_list;
      sample_z_list = sample_y_list;
      sample_yaw_list = sample_x_list;
    }

    int sample_index = 0;
    for (int sample_yaw_idx = 0; sample_yaw_idx < sample_yaw_list.size(); sample_yaw_idx++)
      for (int sample_z_idx = 0; sample_z_idx < sample_z_list.size(); sample_z_idx++)
        for (int sample_y_idx = 0; sample_y_idx < sample_y_list.size(); sample_y_idx++)
          for (int sample_x_idx = 0; sample_x_idx < sample_x_list.size(); sample_x_idx++)
    {
      char sample_index_c[256];
      sprintf(sample_index_c, "%04d", sample_index); // format into 3 digit
      std::cout << "sample_index: " << sample_index << std::endl;
      sample_index++;

      Eigen::Vector3d sample_pose_xyz;
      sample_pose_xyz << sample_x_list(sample_x_idx), sample_y_list(sample_y_idx), sample_z_list(sample_z_idx);

      cuboid *ground_obj = new cuboid();
      Eigen::Matrix<double, 1, 10> measure_data = pred_frame_objects.row(frame_index); // global data
      // read from ground-based data: frameid, xyz, roty, scale, proposal_error
      
      ground_obj->pos = measure_data.block(0,1,1,3).transpose();
      ground_obj->rotY = measure_data(0,4);
      ground_obj->scale = measure_data.block(0,5,1,3).transpose();
      ground_obj->box_corners_3d_world = compute3D_BoxCorner(*ground_obj);    // only need pos, rotY, and scale
      ground_obj->box_corners_2d = compute2D_BoxCorner(*ground_obj, projectionMatrix);// only need corners_3d and projection matrix
      ground_obj->print_cuboid();

      // Eigen::Matrix3d KinvR = Kalib*transToWolrd.topLeftCorner<3, 3>().inverse();
      // double yaw_esti = ground_obj->rotY;
      // Vector2d vp_1 = homo_to_real_coord_vec<double>(KinvR * Vector3d(cos(yaw_esti), sin(yaw_esti), 0));  // for object x axis
      // Vector2d vp_2 = homo_to_real_coord_vec<double>(KinvR * Vector3d(-sin(yaw_esti), cos(yaw_esti), 0)); // for object y axis
      // Vector2d vp_3 = homo_to_real_coord_vec<double>(KinvR * Vector3d(0, 0, 1));                          // for object z axis
      // std::cout << vp_1 << vp_2 << vp_3 << std::endl;
      // sample_obj->skew_ratio = measure_data(0,8);  // temp to be proposal error(maybe not important)
      // sample_obj->print_cuboid();

      cuboid *sample_obj = new cuboid();
      sample_obj->pos = ground_obj->pos + sample_pose_xyz;  
      sample_obj->rotY = ground_obj->rotY + sample_yaw_list(sample_yaw_idx);
      sample_obj->scale = ground_obj->scale;
      sample_obj->box_corners_3d_world = compute3D_BoxCorner(*sample_obj);    // only need pos, rotY, and scale
      sample_obj->box_corners_2d = compute2D_BoxCorner(*sample_obj, projectionMatrix);// only need corners_3d and projection matrix
      sample_obj->skew_ratio = box3d_iou(sample_obj, ground_obj);
      // sample_obj->print_cuboid();


      if (whether_plot_final_images || whether_save_final_images)
      {
        cv::Mat draw_cuboid_img = rgb_img.clone();
        plot_image_with_cuboid_new(draw_cuboid_img, sample_obj);
        if (whether_plot_final_images)
        {
          cv::imshow("draw_cuboid_img", draw_cuboid_img);
          cv::waitKey(0);
        }
        if (whether_save_final_images)
        {
          std::string save_final_image_with_cuboid = save_folder + "image_2/" + frame_index_c + "_" +sample_index_c + "_sample_objects.jpg";
          cv::imwrite( save_final_image_with_cuboid, draw_cuboid_img );
        }
      }

      if (whether_save_cubuid_txt) // save single cubuid
      {
        ofstream sample_cubes;
        string sample_cubes_path = save_folder + "label_2/" + frame_index_c + "_" +sample_index_c + "_sample_objects.txt";
        sample_cubes.open(sample_cubes_path.c_str());
        // Eigen::Matrix<double, 9, 1> save_data;
        // save_data << frame_index, sample_obj->pos, sample_obj->rotY,
        //               sample_obj->scale, sample_obj->skew_ratio; // the last element can be changed to iou
        // sample_cubes << save_data.transpose();
        sample_cubes << frame_index << " " << sample_obj->pos(0) << " " << sample_obj->pos(1) 
            << " " << sample_obj->pos(2) << " " << sample_obj->rotY << " " << sample_obj->scale(0)
            << " " << sample_obj->scale(1) << " " << sample_obj->scale(2) << " " << double(sample_obj->skew_ratio)
            << " " << raw_2d_objs(0,0) << " " << raw_2d_objs(0,1) << " " << raw_2d_objs(0,2) << " " << raw_2d_objs(0,3)
            << " " ;
        sample_cubes.close();
      }    
    }// yolo 2d bounding box
    }// sample loop: x, y, z
  }// sample loop: frame_index






  return 0;
}