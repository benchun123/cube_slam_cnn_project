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

double mse_calculation(const cuboid *sample_obj, const cuboid *ground_obj)
{
    Eigen::Matrix3Xd sample_3d_corners = compute3D_BoxCorner(*sample_obj);    // only need pos, rotY, and scale
    Eigen::Matrix3Xd ground_3d_corners = compute3D_BoxCorner(*ground_obj);    // only need pos, rotY, and scale
    double mse_error = 0;
    for (unsigned corner_idx = 0; corner_idx < 8; corner_idx++)
    {
        double x_err = sample_3d_corners(0,corner_idx) - ground_3d_corners(0,corner_idx);
        double y_err = sample_3d_corners(1,corner_idx) - ground_3d_corners(1,corner_idx);
        double z_err = sample_3d_corners(2,corner_idx) - ground_3d_corners(2,corner_idx);
        double dis_err = sqrt(x_err*x_err + y_err*y_err + z_err*z_err);
        mse_error = mse_error + dis_err;
    }
    mse_error = mse_error / 8;
    return mse_error;
}

double center_orientaion_distance(const cuboid *sample_obj, const cuboid *ground_obj)
{
    Eigen::Vector3d pos1 = sample_obj->pos;
    Eigen::Vector3d pos2 = ground_obj->pos;
    double pose_error = sqrt(pow(pos1(0)-pos2(0),2)+pow(pos1(1)-pos2(1),2)+pow(pos1(2)-pos2(2),2));
    double orientataion_error = sample_obj->rotY - ground_obj->rotY;
    return pose_error+5*orientataion_error;
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
    // ros::init(argc, argv, "detect_3d_cuboid");
    // ros::NodeHandle nh;
    // ca::Profiler::enable();
    string base_folder = argv[1];
    // string dataset_folder = "/home/benchun/benchun/dataset/cubes/training_10/";
    string dataset_folder = base_folder;

    bool whether_plot_final_images = true;
    bool whether_plot_sample_images = false;
    bool whether_plot_detail_images = false;
    bool whether_save_final_images = false;
    bool whether_save_final_cuboids_txt = false;
    bool whether_check_data_balance = false;
    int number_data_balance = 2; // divided distribution (0,1) into 20 pieces, each piece has 2 sample

    std::string truth_camera_pose = base_folder+"online_camera_pose/"+"truth_cam_poses.txt";// data: time, x, y, z, qx, qy, qz, qw
    std::string truth_cube_folder = base_folder+"online_camera_pose/"+"online_cubes.txt";// data: time, x, y, z, qx, qy, qz, qw
    Eigen::MatrixXd truth_frame_poses(100,8);
    Eigen::MatrixXd truth_cube_poses(100,10);
    if (!read_all_number_txt(truth_camera_pose,truth_frame_poses))
	return -1;
    if (!read_all_number_txt(truth_cube_folder,truth_cube_poses))
	return -1;

    int total_frame_number = truth_frame_poses.rows();

    Eigen::Matrix3d Kalib;
    Kalib<< 535.4,  0,  320.1,   // for TUM cabinet data.
            0,  539.2, 247.6,
            0,      0,     1;


    detect_3d_cuboid detect_cuboid_obj;
    detect_cuboid_obj.whether_plot_detail_images = false;
    detect_cuboid_obj.whether_plot_final_images = false;
    detect_cuboid_obj.print_details = false; // false  true
    detect_cuboid_obj.set_calibration(Kalib);
    detect_cuboid_obj.whether_sample_bbox_height = false;
    detect_cuboid_obj.whether_sample_cam_roll_pitch = true;
    detect_cuboid_obj.whether_save_final_cuboids_txt = false;
    detect_cuboid_obj.nominal_skew_ratio = 2;
    detect_cuboid_obj.max_cuboid_num = 1;

    
    Eigen::Matrix<int, 58, 20> data_balance= Eigen::MatrixXi::Zero(58, 20);
    total_frame_number = 1;
    for (int frame_index = 0; frame_index < total_frame_number; frame_index++)
    {
        frame_index = 0;
        bool data_not_full = true;
        // std::cout << "data_balance: \n" << data_balance << std::endl;
        char frame_index_c[256];
        sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit
        std::cout << "frame_index: " << frame_index << std::endl;

        //start detect cuboid
        cv::Mat rgb_img = cv::imread(base_folder+"raw_imgs/"+frame_index_c+"_rgb_raw.jpg", 1);

        //read cleaned yolo 2d object detection
        Eigen::MatrixXd raw_2d_objs(10,5);  // 2d rect [x1 y1 width height], and prob
        raw_2d_objs.setZero();
        if (!read_all_number_txt(base_folder+"/filter_2d_obj_txts/"+frame_index_c+"_yolo2_0.15.txt", raw_2d_objs))
        return -1;
        raw_2d_objs.leftCols<2>().array() -=1;   // change matlab coordinate to c++, minus 1
        // std::cout << "raw_2d_objs" << raw_2d_objs << std::endl;
        if(!raw_2d_objs.isZero())
        {

        if (whether_plot_detail_images)
        {
            Eigen::MatrixXd raw_2d_objs_edge;
            raw_2d_objs_edge.resize(4,4);
            raw_2d_objs_edge << raw_2d_objs(0,0), raw_2d_objs(0,1), raw_2d_objs(0,0)+raw_2d_objs(0,2), raw_2d_objs(0,1),
                                raw_2d_objs(0,0), raw_2d_objs(0,1), raw_2d_objs(0,0), raw_2d_objs(0,1)+raw_2d_objs(0,3),
                                raw_2d_objs(0,0), raw_2d_objs(0,1)+raw_2d_objs(0,3), raw_2d_objs(0,0)+raw_2d_objs(0,2), raw_2d_objs(0,1)+raw_2d_objs(0,3),
                                raw_2d_objs(0,0)+raw_2d_objs(0,2), raw_2d_objs(0,1), raw_2d_objs(0,0)+raw_2d_objs(0,2), raw_2d_objs(0,1)+raw_2d_objs(0,3);
            cv::Mat output_img;
            plot_image_with_edges(rgb_img, output_img, raw_2d_objs_edge, cv::Scalar(255, 0, 0));
            cv::imshow("2d bounding box", output_img);
            cv::waitKey(0);
        }

        //read edge detection
        Eigen::MatrixXd all_lines_raw, edge_txt ;
        if (!read_all_number_txt(base_folder+"online_edge_detection/"+frame_index_c+"_online_edges.txt", edge_txt))
        return -1;
        all_lines_raw.resize(edge_txt.rows(),4);
        all_lines_raw = edge_txt.block(0,0,edge_txt.rows(),4);
        // std::cout << "all_lines_raw: \n" << all_lines_raw << std::endl;

        // only first truth pose is used. to directly visually compare with truth pose. also provide good roll/pitch
        Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
        // Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(0).tail<7>(); // xyz, q1234
        Matrix<double,4,4> transToWolrd;
        transToWolrd.setIdentity();
        transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
        transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
        // std::cout << "transToWolrd: \n" << transToWolrd << std::endl;
        Eigen::Vector3d orientation;
        rot_to_euler_zyx<double>(transToWolrd.block(0,0,3,3), orientation(0), orientation(1), orientation(2));
        std::cout << "camera orientation: " << orientation.transpose() << std::endl;

        ca::Profiler::tictoc("detect_cuboid");
        std::clock_t begin1 = clock();
	    std::vector<ObjectSet> frames_cuboids; // each 2d bbox generates an ObjectSet, which is vector of sorted proposals
        detect_cuboid_obj.detect_cuboid(rgb_img, transToWolrd, raw_2d_objs, all_lines_raw, frames_cuboids);
        ca::Profiler::tictoc("detect_cuboid");
        std::clock_t begin2 = clock();
        std::cout<<"detect_cuboid time: "<< double(begin2-begin1) / CLOCKS_PER_SEC<<std::endl;
	    // if (whether_plot_detail_images)
		// {
		// 	for(int proposal_id=0; proposal_id<frames_cuboids[0].size(); proposal_id++ )
		// 	{
		// 		cv::Mat frame_all_cubes_img = rgb_img.clone();
		// 		std::cout << "proposal: "<<proposal_id<< std::endl;
		// 		plot_image_with_cuboid(frame_all_cubes_img, frames_cuboids[0][proposal_id]);
		// 		cv::imshow("frame_all_cubes_img", frame_all_cubes_img);
		// 		cv::waitKey(0);
		// 	}	
		// }
            
            cuboid *select_cube = frames_cuboids[0][0];
        std::cout << "proposal size: " << frames_cuboids[0].size() << std::endl;
        // to show all the proposal cubes
        for (int cube_idx = 0; cube_idx < frames_cuboids[0].size(); cube_idx++)
        {  
            cuboid *sample_obj = frames_cuboids[0][cube_idx];
            // sample_obj->print_cuboid();
            Eigen::Matrix<double, 1, 10> cube_ground_truth = truth_cube_poses.row(frame_index); // xyz, roty, dimension
            cuboid *ground_obj = new cuboid();
            ground_obj->pos = cube_ground_truth.block(0,1,1,3).transpose();
            ground_obj->rotY = cube_ground_truth(0,4);
            ground_obj->scale = cube_ground_truth.block(0,5,1,3).transpose();
            ground_obj->box_corners_3d_world = compute3D_BoxCorner(*ground_obj);    // only need pos, rotY, and scale
            // if(cube_idx == 0)
            //     ground_obj->print_cuboid();
            sample_obj->skew_ratio = box3d_iou(sample_obj, ground_obj);
            // sample_obj->skew_ratio = center_distance(sample_obj, ground_obj);
            // sample_obj->skew_ratio = mse_calculation(sample_obj, ground_obj);

            if (sample_obj->skew_ratio == 0 )
            {sample_obj->skew_ratio += 0.0001;}
            
            // cuboid selection 
            // std::cout << cube_idx << "proposal_error_new: " << sample_obj->skew_ratio << std::endl;
            if(sample_obj->skew_ratio > select_cube->skew_ratio)
            {
                select_cube = sample_obj;
                std::cout << cube_idx << "select proposal_error_new: " << select_cube->skew_ratio << std::endl;
            }

            char cube_idx_c[256];
            sprintf(cube_idx_c, "%04d", cube_idx); // format into 4 digit

            // plot image and save image
            if (whether_plot_sample_images)
            {
                cv::Mat draw_cuboid_img = rgb_img.clone();
                plot_image_with_cuboid_new(draw_cuboid_img, sample_obj);
                // plot_image_with_cuboid(draw_cuboid_img, sample_obj);
                cv::imshow("draw_cuboid_img", draw_cuboid_img);
                cv::waitKey(0);
            }

            data_not_full = true;
            if(whether_check_data_balance)
            {
                int distribution = floor(sample_obj->skew_ratio * 20); // floor(1.2) = 1
                data_balance(frame_index, distribution) +=1;
                if (data_balance(frame_index, distribution) > number_data_balance)
                    data_not_full = false;
            } 

            // if save single cubes and camera poses
            if (whether_save_final_images && whether_save_final_cuboids_txt && data_not_full)
            {

                std::cout << cube_idx<<"data_balance" << data_balance.row(frame_index) << std::endl;

                // save sample image
                cv::Mat save_cuboid_img = rgb_img.clone();
                plot_image_with_cuboid_new(save_cuboid_img, sample_obj);
                std::string save_sample_image_path = dataset_folder +"image_2/" + frame_index_c + 
                                                           "_" + cube_idx_c +"_sample_proposal.jpg";
                cv::imwrite( save_sample_image_path, save_cuboid_img );
                
                // save cuboid
                ofstream cube_pose_data;
                string cube_pose_path = dataset_folder +"label_2/" + frame_index_c + 
                                        "_" + cube_idx_c +"_sample_proposal.txt";
                cube_pose_data.open(cube_pose_path.c_str());
                // Eigen::Matrix<double, 9, 1> cube_pose;
                // cube_pose << frame_index, sample_obj->pos, sample_obj->rotY,
                //             sample_obj->scale, sample_obj->skew_ratio;
                // cube_pose_data << cube_pose.transpose();
                cube_pose_data << frame_index << " " << sample_obj->pos(0) << " " << sample_obj->pos(1) 
                    << " " << sample_obj->pos(2) << " " << sample_obj->rotY << " " << sample_obj->scale(0)
                    << " " << sample_obj->scale(1) << " " << sample_obj->scale(2) << " " << double(sample_obj->skew_ratio)
                    << " " << raw_2d_objs(0,0) << " " << raw_2d_objs(0,1) << " " << raw_2d_objs(0,2) << " " << raw_2d_objs(0,3)
                    << " " ;
                cube_pose_data.close();   

                // // save camera
                // ofstream cube_camera_data;
                // string cube_camera_path = dataset_folder +"camera_2/" +frame_index_c + 
                //                         "_" + cube_idx_c +"_sample_camera.txt";
                // cube_camera_data.open(cube_camera_path.c_str());
                // Vector3d new_camera_eulers =  detect_cuboid_obj.cam_pose_raw.euler_angle;
                // new_camera_eulers(0) += sample_obj->camera_roll_delta; new_camera_eulers(1) += sample_obj->camera_pitch_delta;
                // Matrix3d rotation_new = euler_zyx_to_rot<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
                // Vector3d trans = transToWolrd.col(3).head<3>();
                // Eigen::Quaterniond qwxyz = zyx_euler_to_quat<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
                // Eigen::Matrix<double, 8, 1> new_camera_pose;
                // new_camera_pose << frame_index, trans, qwxyz.x(), qwxyz.y(), qwxyz.z(), qwxyz.w();
                // cube_camera_data << new_camera_pose.transpose() <<"\n";
                // cube_camera_data.close();
            }

        }
        // plot every frame
        if (whether_plot_final_images)
        {
            select_cube->print_cuboid();
            std::cout << "proposal_error_new: " << select_cube->skew_ratio << std::endl;
            cv::Mat draw_cuboid_img = rgb_img.clone();
            plot_image_with_cuboid_new(draw_cuboid_img, select_cube);
            // plot_image_with_cuboid_new(draw_cuboid_img, frames_cuboids[0][1607]);
            cv::imshow("image every frame", draw_cuboid_img);
            cv::waitKey(0);
        }

        // // // save all cubes and camera poses
        // if(whether_save_final_cuboids_txt)
        // {
        //      cuboid *detected_cube = select_cube;
        // 	Eigen::Matrix<double, 9, 1> cube_pose;
        // 	cube_pose<<frame_index,detected_cube->pos(0),detected_cube->pos(1),detected_cube->pos(2),detected_cube->rotY,
        // 					detected_cube->scale(0),detected_cube->scale(1),detected_cube->scale(2), detected_cube->skew_ratio;  // xyz roll pitch yaw scale
        // 	online_stream_cube << cube_pose.transpose() <<"\n";
        //     // online_stream_cube << frame_index << " " << detected_cube->pos(0) << " " << detected_cube->pos(1) 
        //     //     << " " << detected_cube->pos(2) << " " << detected_cube->rotY << " " << detected_cube->scale(0)
        //     //     << " " << detected_cube->scale(1) << " " << detected_cube->scale(2) << " " << double(detected_cube->skew_ratio)
        //     //     << " " << raw_2d_objs(0,0) << " "  << raw_2d_objs(0,1) << " " << raw_2d_objs(0,2) << " " << raw_2d_objs(0,3)
        //     //     << " " << frames_cuboids[0].size() << " " << selected_cube_idx << " " <<"\n" ;
        //     Vector3d new_camera_eulers =  detect_cuboid_obj.cam_pose_raw.euler_angle;
        //     new_camera_eulers(0) += detected_cube->camera_roll_delta; new_camera_eulers(1) += detected_cube->camera_pitch_delta;
        //     Matrix3d rotation_new = euler_zyx_to_rot<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
        //     Vector3d trans = transToWolrd.col(3).head<3>();
        //     Eigen::Quaterniond qwxyz = zyx_euler_to_quat<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
        //     Eigen::Matrix<double, 8, 1> new_camera_pose;
        //     new_camera_pose << frame_index, trans, qwxyz.x(), qwxyz.y(), qwxyz.z(), qwxyz.w();
        // 	online_stream_camera << new_camera_pose.transpose() <<"\n";
        // }
        }
    }



    return 0;
}