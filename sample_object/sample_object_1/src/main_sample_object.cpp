// std c
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// ours
#include "sample_object/sample_object.h"
// boost and geometry, main for polygon
#include <deque>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace bg = boost::geometry;

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char **argv)
{

  bool whether_plot_final_images = true;
  bool whether_save_final_images = false;
  bool whether_save_cubuid_txt = false;
  bool whether_sample_images = false;
  bool other_function_test = false;

  string base_folder = argv[1];
  string imgs_folder = base_folder + "/raw_imgs/";
  string save_folder = "/home/benchun/benchun/dataset/cubes/training_2/";

  // std::string pred_objs_txt = base_folder + "0000_3d_cuboids.txt"; // data: x y z yaw length width height
  std::string pred_objs_txt = base_folder+"online_cubes.txt";  // data: x, y, z, pitch, toll, yaw, length, width, height
  std::string init_camera_pose = base_folder+"online_camera.txt"; // // time x y z qx qy qz qw
  std::string truth_camera_pose = base_folder+"truth_cam_poses.txt";  // data: x, y, z, qx, qy, qz, qw
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

  // // parameters for test image chair
  // Matrix3d Kalib;
  // Kalib << 529.5000, 0, 365.0000,
  //     0, 529.5000, 265.0000,
  //     0, 0, 1.0000;
  // Matrix4d transToWolrd;
  // transToWolrd << 1, 0.0011, 0.0004, 0, // hard coded  NOTE if accurate camera roll/pitch, could sample it!
  //     0, -0.3376, 0.9413, 0,
  //     0.0011, -0.9413, -0.3376, 1.35,
  //     0, 0, 0, 1;
  // cam_pose_infos cam_pose;
  // cam_pose.Kalib = Kalib;
  // cam_pose.invK = Kalib.inverse();
  // cam_pose.transToWolrd = transToWolrd;
	// cam_pose.rotationToWorld = transToWolrd.topLeftCorner<3, 3>();
	// Vector3d euler_angles;
	// quat_to_euler_zyx(Quaterniond(cam_pose.rotationToWorld), euler_angles(0), euler_angles(1), euler_angles(2));
	// cam_pose.euler_angle = euler_angles;
	// cam_pose.invR = cam_pose.rotationToWorld.inverse();
	// cam_pose.projectionMatrix = cam_pose.Kalib * transToWolrd.inverse().topRows<3>(); // project world coordinate to camera
	// cam_pose.KinvR = cam_pose.Kalib * cam_pose.invR;
	// cam_pose.camera_yaw = cam_pose.euler_angle(2)

  // sample_obj->box_corners_3d_world = compute3D_BoxCorner(*sample_obj);    // only need pos, rotY, and scale
  // sample_obj->box_corners_2d = compute2D_BoxCorner(*sample_obj, projectionMatrix);// only need corners_3d and projection matrix

    // // a ground truth for debugging
    // printing cuboids info....
    // pos   -0.226672   1.72969   0.46817
    // scale   0.0951803  0.344872   0.46817
    // rotY   -0.785451
    // box_config_type   1 1
    // box_corners_2d
    // 384 245 216 353 388 220 187 352
    // 412 499 470 394 195 232 220 188
    // box_corners_3d_world
    //   0.084501  -0.403247  -0.537846 -0.0500972   0.084501  -0.403247  -0.537846 -0.0500972
    //    1.90624    1.41854    1.55315    2.04085    1.90624    1.41854    1.55315    2.04085
    //          0          0          0          0    0.93634    0.93634    0.93634    0.93634

  Eigen::Matrix3d Kalib;
  Kalib<< 535.4,  0,  320.1,   // for TUM cabinet data.
          0,  539.2, 247.6,
          0,      0,     1;

  total_frame_number = 4;
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
    sample_x_list.resize(1,9);
    sample_x_list << 0, -0.03, 0.03, -0.06, 0.06, -0.09, 0.09, -0.12, 0.12;
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

      // // 2d ground truth read from global txt
      // sample_obj->pos= Eigen::Vector3d(measure_data(0), measure_data(1), measure_data(2));
      // sample_obj->rotY= measure_data(5);
      // sample_obj->scale= Eigen::Vector3d(measure_data(6), measure_data(7), measure_data(8));

      // // from global to local
			// Matrix<double,4,4> cube_global_pose;
			// cube_global_pose.setIdentity();
			// cube_global_pose.block(0,0,3,3) = euler_zyx_to_rot(measure_data(3),measure_data(4),measure_data(5));
			// cube_global_pose.col(3).head(3) = Eigen::Vector3d(measure_data(0), measure_data(1), measure_data(2));
      // Matrix<double,4,4>  cube_local_pose = transToWolrd.inverse() * cube_global_pose;
			// Eigen::Vector3d alpha;
			// alpha(0) = atan2(cube_local_pose(2,1), cube_local_pose(2,2));
			// alpha(1) = atan2(-cube_local_pose(2,0), sqrt(cube_local_pose(2,1)*cube_local_pose(2,1) + cube_local_pose(2,2)*cube_local_pose(2,2)));
			// alpha(2) = atan2(cube_local_pose(1,0), cube_local_pose(0,0));
			// Eigen::Matrix<double, 9, 1> cube_pose;
			// cube_pose << cube_local_pose.col(3).head(3), alpha, measure_data(6), measure_data(7), measure_data(8);
			// std::cout << cube_pose.transpose() << std::endl;

      // // from local to ground pose
			// // read ground-based ground truth of previous objets
			// Matrix<double,4,4> cube_ground_pose;
      // cube_ground_pose = transToWolrd * cube_local_pose;
			// Eigen::Vector3d beta;
			// beta(0) = atan2(cube_ground_pose(2,1), cube_ground_pose(2,2));
			// beta(1) = atan2(-cube_ground_pose(2,0), sqrt(cube_ground_pose(2,1)*cube_ground_pose(2,1) + cube_ground_pose(2,2)*cube_ground_pose(2,2)));
			// beta(2) = atan2(cube_ground_pose(1,0), cube_ground_pose(0,0));
			// Eigen::Matrix<double, 9, 1> cube_ground_pose_vector;
			// cube_ground_pose_vector << cube_ground_pose.col(3).head(3), beta, measure_data(6), measure_data(7), measure_data(8);
			// std::cout << cube_ground_pose_vector.transpose() << std::endl;

      // Matrix<double,4,4> cube_global_pose, cube_local_pose;
      // cube_local_pose.setIdentity();
      // cube_local_pose.block(0,0,3,3) = euler_zyx_to_rot(measure_data(3),measure_data(4),measure_data(5));
      // cube_local_pose.col(3).head(3) = Eigen::Vector3d(measure_data(0), measure_data(1), measure_data(2));
      // cube_global_pose = transToWolrd * cube_local_pose;
      // sample_obj->pos= cube_global_pose.col(3).head(3);
      // sample_obj->rotY= atan2(cube_global_pose(1,0), cube_global_pose(0,0));
      // sample_obj->scale= measure_data.block(0,6,1,3).transpose();
      // sample_obj->print_cuboid();
      // std::cout << "local_pose: " << measure_data << std::endl;
      // std::cout << "global_pose: " << cube_global_pose.col(3).head(3) << " " << sample_obj->rotY << std::endl;
      // std::cout << "transToWolrd: " << transToWolrd << std::endl;

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
      // sample_obj->pos= sample_obj->pos + Eigen::Vector3d(-0.07, 0.02, 0);   // manual correct for ground truth.
      // sample_obj->rotY= sample_obj->rotY - 0.0 ;
      // sample_obj->scale= sample_obj->scale + Eigen::Vector3d(-0.01, -0.03, -0.0);

      // get 3d corner from position, rotation and scale, project 3d corners to 2d image points
      // cuboid *sample_obj = new cuboid();
      // sample_obj->pos = Vector3d(-1.52693,   0.461311,   0.280412);
      // sample_obj->rotY = 3.06922;
      // sample_obj->scale = Vector3d(0.393607,   0.242375,   0.261626); // length, width, height
      // Eigen::Matrix<double, 3, 8> corner_3d_tmp;
      // corner_3d_tmp <<
      //   -1.95923,  -1.92322,  -1.09305,  -1.12906,  -1.96077,  -1.92476,  -1.09459,   -1.1306,
      //   0.247057,  0.744004,  0.683829,  0.186883,  0.238777,  0.735724,   0.67555,  0.178603,
      //   0.0112438, 0.0191288, 0.0206104, 0.0127254,  0.540127,  0.548012,  0.549493,  0.541608;
      // sample_obj->box_corners_3d_world = corner_3d_tmp;

      sample_obj->box_corners_3d_world = compute3D_BoxCorner(*sample_obj);    // only need pos, rotY, and scale
      sample_obj->box_corners_2d = compute2D_BoxCorner(*sample_obj, projectionMatrix);// only need corners_3d and projection matrix
      sample_obj->skew_ratio = box3d_iou(sample_obj, ground_obj);
      // sample_obj->print_cuboid();

      // // from 2d to 3d, cubeslam, use ground plane, calculate scale, orientation and position first and expand to 3d cube
      // cuboid *sample_obj = new cuboid();
      // Eigen::MatrixXd box_corners_2d(2,8);
      // box_corners_2d <<   207,     182, 440.079,     465,  408.72, 209.309, 228.853, 428.292,
      //                     37, 44.7421, 163.019, 146.882,     350, 208.064, 197.904, 332.048;
      // Eigen::Vector3d box_config_type(1, 2, -4.36615); // config determines the corner order.
      // Vector4d ground_plane_world(0, 0, 1, 0); // treated as column vector % in my pop-up code, I use [0 0 -1 0]. here I want the normal pointing innerwards, towards the camera to match surface normal prediction
	    // Vector4d ground_plane_sensor = transToWolrd.transpose() * ground_plane_world;
      // Eigen::Matrix3d invK = Kalib.inverse();
      // change_2d_corner_to_3d_object(box_corners_2d, box_config_type, ground_plane_sensor, transToWolrd, invK, projectionMatrix, *sample_obj);
      // std::cout << "ground_plane_sensor: \n" << ground_plane_sensor.transpose() << std::endl;
      // std::cout << "invK: \n" << invK << std::endl;
      // std::cout << "projectionMatrix: \n" << projectionMatrix << std::endl;
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

      // if (whether_save_cubuid_txt) // transform from camera to local to realtive ground valu
      // {
      //   Eigen::MatrixXd temp_objects(100,10);  // 100 is some large row number, each row in txt has 10 numbers
      //   char frame_index_c[256];
      //   sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit
      //   string initialpath = base_folder + "/training/previous_obj_poses_gt_global.txt";
      //   string finalpath = base_folder + "/training/prevous_obj_poses_temp.txt";
      //   if (!read_all_number_txt(initialpath, temp_objects))
      //     return -1;
      //   ofstream groundvalue;
      //   groundvalue.open(finalpath.c_str());
      //   for (size_t frame_index = 0; frame_index < 58; frame_index++)
      //   {
      //     Eigen::Matrix<double, 1, 9> measure_data = temp_objects.row(frame_index);
      //     // measure_data(0)=measure_data(0)-0.08;
      //     // measure_data(1)=measure_data(1)+0.06;
      //     // measure_data(2)=measure_data(2)+0.02;
      //     // measure_data(5)=measure_data(5)-0.05;
      //     // from global to local pose
      //     Matrix<double,4,4> cube_global_pose, cube_local_pose, cube_grond_value;
      //     cube_global_pose.setIdentity();
      //     cube_global_pose.block(0,0,3,3) = euler_zyx_to_rot(measure_data(3),measure_data(4),measure_data(5));
      //     cube_global_pose.col(3).head(3) = Eigen::Vector3d(measure_data(0), measure_data(1), measure_data(2));
      //     Matrix<double,4,4> transToWolrd;
      //     Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
      //     transToWolrd.setIdentity();
      //     transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
      //     transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
      //     cube_local_pose = transToWolrd.inverse() * cube_global_pose;
      //     Eigen::Vector3d alpha;
      //     alpha(0) = atan2(cube_local_pose(2,1), cube_local_pose(2,2));
      //     alpha(1) = atan2(-cube_local_pose(2,0), sqrt(cube_local_pose(2,1)*cube_local_pose(2,1) + cube_local_pose(2,2)*cube_local_pose(2,2)));
      //     alpha(2) = atan2(cube_local_pose(1,0), cube_local_pose(0,0));
      //     Eigen::Matrix<double, 9, 1> output_local;
      //     output_local << cube_local_pose.col(3).head(3), alpha, measure_data(6), measure_data(7), measure_data(8);
      //     std::cout << "local_pose: " << output_local.transpose() << std::endl;
      //     groundvalue << output_local.transpose()  <<"\n";

      //     // from local pose to relative ground value
      //     Eigen::Matrix<double, 1, 7> cam_pose_vec = init_frame_poses.row(frame_index).segment<7>(1);// time x y z qx qy qz qw
      //     // Quaterniond notknow=Quaterniond(cam_pose_vec(6),cam_pose_vec(3),cam_pose_vec(4),cam_pose_vec(5));
      //     // if (notknow.w()<0){
      //     //     notknow.coeffs() *= -1;
      //     // }
      //     // notknow.normalize();
      //     Matrix<double,4,4> cam_val_Twc;
      //     cam_val_Twc.setIdentity();
      //     cam_val_Twc.block(0,0,3,3) = Quaterniond(cam_pose_vec(6),cam_pose_vec(3),cam_pose_vec(4),cam_pose_vec(5)).toRotationMatrix();
      //     // cam_val_Twc.block(0,0,3,3) = notknow.toRotationMatrix();
      //     cam_val_Twc.col(3).head(3) = Eigen::Vector3d(cam_pose_vec(0), cam_pose_vec(1), cam_pose_vec(2));
      //     cube_grond_value = cam_val_Twc * cube_local_pose;
      //     Eigen::Matrix3d rotation_matrix = cube_grond_value.block(0,0,3,3);
      //     Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles ( 0,1,2 ); // rotation matrix to eulerangle, xyz
      //     Eigen::Matrix<double, 9, 1> output_ground_value;
      //     output_ground_value << frame_index, cube_grond_value.col(3).head(3), euler_angles(2), measure_data(6), measure_data(7), measure_data(8),0.26;
      //     // std::cout << "output_ground_value" << output_ground_value.transpose()  << std::endl;
      //     // groundvalue << output_final.transpose()  <<"\n";
      //   }
      //   groundvalue.close();
      // }
    
    }// yolo 2d bounding box
    }// sample loop: x, y, z
  }// sample loop: frame_index

  if(other_function_test)
  {
    std::cout << "iou calculation test: " << std::endl;
    cuboid *sample_obj = new cuboid();
    sample_obj->pos = Eigen::Vector3d(2.882992 ,1.698800 ,20.785644);
    sample_obj->rotY =  -1.531692;
    sample_obj->scale = Eigen::Vector3d(1.497255/2,1.644981/2, 3.628938/2);
    sample_obj->box_corners_3d_world = compute3D_BoxCorner(*sample_obj);    // only need pos, rotY, and scale
    // sample_obj->print_cuboid();
    cuboid *ground_obj = new cuboid();
    ground_obj->pos = Eigen::Vector3d(2.756923, 1.661275, 20.943280);
    ground_obj->rotY = -1.549553;
    ground_obj->scale = Eigen::Vector3d(1.458242/2, 1.604773/2, 3.707947/2);
    ground_obj->box_corners_3d_world = compute3D_BoxCorner(*ground_obj);    // only need pos, rotY, and scale
    // ground_obj->print_cuboid();
    // double box_iou_3d=box3d_iou(sample_obj, ground_obj);
    // std::cout << "box_iou_3d: " << box_iou_3d<< std::endl;

    // get 2d area in the top
    // append outer and inners() https://www.boost.org/doc/libs/1_65_1/libs/geometry/doc/html/geometry/reference/models/model_polygon.html
    typedef bg::model::point<double, 2, bg::cs::cartesian> point_t;
    typedef bg::model::polygon<point_t> polygon_t;
    polygon_t sam_poly, gt_poly;

    Eigen::Matrix<double, 4, 2> sam_corners, gt_corners;
    for (size_t i = 0; i < 5; i++)
    {
      if (i == 4) // final number, make a circle
      {
      point_t sam_top_points(sample_obj->box_corners_3d_world(0,0),sample_obj->box_corners_3d_world(1,0));
      boost::geometry::append(sam_poly.outer(), sam_top_points);
      point_t gt_top_points(ground_obj->box_corners_3d_world(0,0),ground_obj->box_corners_3d_world(1.0));
      boost::geometry::append(gt_poly.outer(), gt_top_points);    
      }
      else
      {
      point_t sam_top_points(sample_obj->box_corners_3d_world(0,i),sample_obj->box_corners_3d_world(1,i));
      boost::geometry::append(sam_poly.outer(), sam_top_points);
      point_t gt_top_points(ground_obj->box_corners_3d_world(0,i),ground_obj->box_corners_3d_world(1,i));
      boost::geometry::append(gt_poly.outer(), gt_top_points);
      }
    }

    std::cout << "sam_poly: " << bg::area(sam_poly)<< std::endl;
    std::cout << "gt_poly: " << bg::area(gt_poly)<< std::endl;

    std::vector<polygon_t> inter_poly;
    boost::geometry::intersection(sam_poly, gt_poly, inter_poly);
    // boost::geometry::union_(poly1, poly2, un);
    double inter_area = inter_poly.empty() ? 0 : boost::geometry::area(inter_poly.front());
    double union_area = bg::area(sam_poly) + bg::area(gt_poly) - inter_area;
    double iou_2d = inter_area / union_area;
    std::cout << "inter_area: " << inter_area<< std::endl;
    std::cout << "iou2d: " << iou_2d << std::endl;

    double h_up = min(sample_obj->box_corners_3d_world(2,4),ground_obj->box_corners_3d_world(2,4));
    double h_down = max(sample_obj->box_corners_3d_world(2,0),ground_obj->box_corners_3d_world(2,0));
    double inter_vol = inter_area * max(0.0, h_up - h_down);
    double sam_vol = sample_obj->scale(0)*2 * sample_obj->scale(1)*2 * sample_obj->scale(2)*2;
    double gt_vol = ground_obj->scale(0)*2 * ground_obj->scale(1)*2 * ground_obj->scale(2)*2;
    double iou_3d = inter_vol / (sam_vol + gt_vol - inter_vol);
    std::cout << "h_up: " << h_up << " h_down: " << h_down<< std::endl;
    std::cout << "sam_vol: " << sam_vol << " gt_vol: " << gt_vol<< std::endl;
    std::cout << "inter_vol: " << inter_vol << std::endl;
    std::cout << "iou3d: " << iou_3d << std::endl;

  }




  return 0;
}