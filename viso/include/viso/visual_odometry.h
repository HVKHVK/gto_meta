#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_

#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <pwd.h>
#include <fstream>
#include <queue>
#include <mutex>
#include <boost/circular_buffer.hpp>

#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>

#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cv_bridge/cv_bridge.h>

#include <dynamic_reconfigure/DoubleParameter.h>
#include <dynamic_reconfigure/Reconfigure.h>
#include <dynamic_reconfigure/Config.h>
#include <dynamic_reconfigure/server.h>

#include <gto/gtoConfig.h>
#include <gto/PoseCmd.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>

namespace visual_odometry
{

  class VisualOdometry
  {

    public:
        VisualOdometry();
        ~VisualOdometry();
        void main();

    private:        
        void imageCallback(const sensor_msgs::CompressedImage::ConstPtr& comp_image);
        void infoCallback(const sensor_msgs::CameraInfo::ConstPtr& cam_info);
        void initialFrame();
        void process();
        void paramCallback(gto::gtoConfig &config, uint32_t level);
        void publishOdom();
        void matchFrame(cv::Mat& secundum_frame, std::vector<cv::KeyPoint>& secundum_keypoints, cv::Mat& secundum_descriptors);
        bool odomResetCallback(gto::PoseCmd::Request &req , gto::PoseCmd::Response &res);

        sensor_msgs::Image cvToRosImage(const cv::Mat frame );

        ros::NodeHandle private_nh_;
        ros::NodeHandle nh_;
        
        ros::Rate rate;

        ros::Subscriber image_subscriber_;
        ros::Subscriber camera_info_;
        ros::Publisher output_odom_;

        ros::ServiceServer odom_reset_;
        // Debug Publishers
        ros::Publisher keypoint_publisher_;
        ros::Publisher matcher_publisher_;
        ros::Publisher selected_keypoint_publisher_;
        ros::Publisher selected_keypoint_matcher_publisher_;
      
        boost::circular_buffer<cv::Mat> image_queue_;

        cv::Ptr<cv::ORB> orb;
        cv::Ptr<cv::Feature2D> detector;
        cv::Ptr<cv::DescriptorMatcher> matcher;

        cv::TickMeter timer;

        cv::Mat primum_frame;
        cv::Mat primum_descriptors;
        std::vector<cv::KeyPoint> primum_keypoints;

        dynamic_reconfigure::Server<gto::gtoConfig> *dsrv_;
        bool debug_;
        double nn_match_ratio, matcher_radius, homography_ransac_threshold, homography_confidence;
        int matcher_type, matcher_knn_k, homography_method, homography_max_iter, normal_axis;
        
        boost::array<double, 9UL> K;
        std::vector<double> D;

        double factor_d1, normals_sublimit;

        nav_msgs::Odometry msgs;
        double x = 0;
        double y = 0;
        double z = 0;
        double roll = 0;
        double pitch = 0;
        double yaw = 0;

        std::mutex mutex;

        tf2_ros::TransformBroadcaster tf_br;

        int camera_w_px = 352;
        int camera_h_px = 240;
        double camera_w_mm = 75;
        double camera_h_mm = 37;
  };
};
#endif // VISUAL_ODOMETRY_H_