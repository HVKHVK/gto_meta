#include <gto/ground_texture_odometry.h>

namespace ground_texture_odometry{

  GroundTextureOdometry::GroundTextureOdometry(): private_nh_("~"), rate(100)
  {

    image_subscriber_ = nh_.subscribe("image_raw", 100, &GroundTextureOdometry::imageCallback, this);
    camera_info_ = nh_.subscribe("camera_info", 10, &GroundTextureOdometry::infoCallback, this);
    output_odom_ = private_nh_.advertise<nav_msgs::Odometry>("odom", 10, true);

    keypoint_publisher_ = private_nh_.advertise<sensor_msgs::Image>("I_keypoint_publisher", 10, true);
    matcher_publisher_ = private_nh_.advertise<sensor_msgs::Image>("II_matcher_publisher", 10, true);
    selected_keypoint_publisher_ = private_nh_.advertise<sensor_msgs::Image>("III_selected_keypoint_publisher", 10, true);
    selected_keypoint_matcher_publisher_ = private_nh_.advertise<sensor_msgs::Image>("IV_selected_keypoint_matcher_publisher", 10, true);

    orb = cv::ORB::create();
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    dsrv_ = new dynamic_reconfigure::Server<gto::gtoConfig>(ros::NodeHandle("~"));
    dynamic_reconfigure::Server<gto::gtoConfig>::CallbackType cb = boost::bind(&GroundTextureOdometry::paramCallback, this, _1, _2);
    dsrv_->setCallback(cb);    
    
    odom_reset_ = private_nh_.advertiseService("odom_reset", &GroundTextureOdometry::odomResetCallback, this);
    //private_nh_.param("debug", debug_, true);
    image_queue_ = boost::circular_buffer<cv::Mat>(100);

    msgs.header.frame_id = "map";
    msgs.child_frame_id = "odom";

    while(image_queue_.empty())
    {
      ROS_ERROR_THROTTLE(1, "No input Image.");
      rate.sleep();      
      ros::spinOnce();
    }
    ROS_INFO("Initializing...");

    GroundTextureOdometry::initialFrame();

    ROS_INFO("Starting...");
  };

  GroundTextureOdometry::~GroundTextureOdometry()
  {
  };

  void GroundTextureOdometry::main()
  {
    while(ros::ok())
    {
      timer.start();
      GroundTextureOdometry::process();
      rate.sleep();      
      timer.stop();
      ROS_INFO("FPS: (%f)", 1.0 / timer.getTimeSec());
      timer.reset();
      ros::spinOnce();
    }
  }
  
  bool GroundTextureOdometry::odomResetCallback(gto::PoseCmd::Request &req , gto::PoseCmd::Response &res)
  {
    std::lock_guard<std::mutex> lock(mutex);
    x = req.pose.position.x;
    y = req.pose.position.y;
    return true;
  }

  void GroundTextureOdometry::process()
  {
    std::lock_guard<std::mutex> lock(mutex);

    if(image_queue_.empty())
    {
      ROS_ERROR("Image Queue Empty");
      return;
    }
    cv::Mat secundum_frame = image_queue_.front();
    image_queue_.pop_front();

    cv::Mat secundum_descriptors;
    std::vector<cv::KeyPoint> secundum_keypoints;
    orb->detectAndCompute(secundum_frame, cv::noArray(), secundum_keypoints, secundum_descriptors);

    // Show Keypoints of Frame (I)
    if (debug_)
    {
      cv::Mat keypoint_frame;
      cv::drawKeypoints( secundum_frame, secundum_keypoints, keypoint_frame, cv::Scalar( 0, 255, 0 ));
      keypoint_publisher_.publish(GroundTextureOdometry::cvToRosImage(keypoint_frame));
    }

    std::vector< std::vector<cv::DMatch>> matches;
    try
    {
      if (matcher_type == 0)
        matcher->knnMatch(primum_descriptors, secundum_descriptors, matches, matcher_knn_k);
      else if (matcher_type == 1)
        matcher->radiusMatch(primum_descriptors, secundum_descriptors, matches, matcher_radius);
      else
        ROS_ERROR("Matcher Type Error");
    }catch (cv::Exception& e)
    {
      ROS_ERROR("Matcher: ", e.what());
      GroundTextureOdometry::matchFrame(secundum_frame, secundum_keypoints, secundum_descriptors);
      return;
    }

    // Show Matches (II)
    if (debug_)
    {
      try
      {
        cv::Mat match_frame;
        cv::drawMatches(primum_frame, primum_keypoints, secundum_frame, secundum_keypoints, matches, match_frame, cv::Scalar( 255, 0, 0 ), cv::Scalar( 255, 0, 0 ));
        matcher_publisher_.publish(GroundTextureOdometry::cvToRosImage(match_frame));
      } catch (std::exception& e)
      {
        ROS_ERROR("Debug II: ", e.what());
        return;
      }
    }
      
    std::vector<cv::KeyPoint> primum_matched_keypoints, secundum_matched_keypoints; 
    std::vector<cv::DMatch> filtered_matches;

    try
    {
      int counter = 0;
      for(int i = 0; i < matches.size(); i++) 
      {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance)
        {
          primum_matched_keypoints.push_back(primum_keypoints[matches[i][0].queryIdx]);
          secundum_matched_keypoints.push_back(secundum_keypoints[matches[i][0].trainIdx]);
          filtered_matches.push_back(cv::DMatch(counter,counter,0));
          counter++;
        }   
      }
    }catch (std::exception& e)
    {
      ROS_ERROR("Matcher Filter: ", e.what());
      return;
    }

    // Show Selected Keypoints (III)
    if (debug_)
    {
      //cv::Mat primum_keypoint_frame, secundum_keypoint_frame, concat_frame;
      //cv::drawKeypoints( primum_frame, primum_matched_keypoints, primum_keypoint_frame, cv::Scalar( 0, 0, 200 ));
      //cv::drawKeypoints( secundum_frame, secundum_matched_keypoints, secundum_keypoint_frame, cv::Scalar( 0, 0, 200 ));
      //cv::hconcat(primum_keypoint_frame, secundum_keypoint_frame, concat_frame);
      //selected_keypoint_publisher_.publish(GroundTextureOdometry::cvToRosImage(concat_frame));

      cv::Mat concat_frame;
      cv::drawMatches(primum_frame, primum_matched_keypoints, secundum_frame, secundum_matched_keypoints, filtered_matches, concat_frame, cv::Scalar( 0, 0, 255 ), cv::Scalar( 0, 0, 255 ));
      selected_keypoint_publisher_.publish(GroundTextureOdometry::cvToRosImage(concat_frame));
    }

    cv::Mat homography, inlier_mask, test;
    if(primum_matched_keypoints.size() >= 4) 
    {

      std::vector<cv::Point2f> primum_matched_points, secundum_matched_points;
    
      cv::KeyPoint::convert(primum_matched_keypoints, primum_matched_points);
      cv::KeyPoint::convert(secundum_matched_keypoints, secundum_matched_points);

      int method;
      if (homography_method == 0)
        method = 0;
      else if (homography_method == 1)
        method = cv::RANSAC;
      else if (homography_method == 2)
        method = cv::LMEDS;
      else if (homography_method == 3)
        method = cv::RHO;
      else 
        ROS_ERROR("Homography Method Not Valid");

      homography = cv::findHomography(primum_matched_points, secundum_matched_points,
                                      method, homography_ransac_threshold,
                                      inlier_mask, homography_max_iter,
                                      homography_confidence);
    }

    if(primum_matched_keypoints.size() < 4 || homography.empty()) {
      ROS_ERROR("Homography Empty");
      GroundTextureOdometry::matchFrame(secundum_frame, secundum_keypoints, secundum_descriptors);
      return;
    }
    
    std::vector<cv::KeyPoint> primum_inliers, secundum_inliers;
    std::vector<cv::DMatch> inlier_matches;
    for(unsigned i = 0; i < primum_matched_keypoints.size(); i++) 
    {
      if(inlier_mask.at<uchar>(i)) 
      {
        int new_i = static_cast<int>(primum_inliers.size());
        primum_inliers.push_back(primum_matched_keypoints[i]);
        secundum_inliers.push_back(secundum_matched_keypoints[i]);
        inlier_matches.push_back(cv::DMatch(new_i, new_i, 0));
      }
    }

    // Show Selected Keypoints Match (IV)
    if (debug_)
    {
      cv::Mat res;
      cv::drawMatches(primum_frame, primum_inliers, secundum_frame, secundum_inliers, inlier_matches, res, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
      selected_keypoint_matcher_publisher_.publish(GroundTextureOdometry::cvToRosImage(res));
    }

    std::vector<double> x_diff, y_diff;
    for (int i = 0; i < primum_inliers.size(); i++)
    {
      x_diff.push_back((camera_w_mm / camera_w_px) * (secundum_inliers[i].pt.x - primum_inliers[i].pt.x));
      y_diff.push_back((camera_h_mm / camera_h_px) * (secundum_inliers[i].pt.y - primum_inliers[i].pt.y));

      //ROS_DEBUG_STREAM("X_diff: " << x_diff[i]);
      //ROS_DEBUG_STREAM("Y_diff: " << y_diff[i]);
    }

    double x_mean = std::accumulate(x_diff.begin(), x_diff.end(), 0.0) / x_diff.size();
    double y_mean = std::accumulate(y_diff.begin(), y_diff.end(), 0.0) / y_diff.size();

    ROS_DEBUG_STREAM("X_mean: " << x_mean);
    ROS_DEBUG_STREAM("Y_mean: " << y_mean);

    std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;
    int solutions;
    try
    {    
      cv::Mat cameraMatrix(3, 3, CV_64F, &K);
      ROS_DEBUG_STREAM("Camera Calibration " << cameraMatrix );
      solutions = cv::decomposeHomographyMat(homography, cameraMatrix, Rs_decomp, ts_decomp, normals_decomp);
    }catch (cv::Exception& e)
    {
      ROS_ERROR("Decompose: ", e.what());
      GroundTextureOdometry::matchFrame(secundum_frame, secundum_keypoints, secundum_descriptors);
      return;
    }
  
    int index = 0;
    double base = 0;
    for(int i = 0; i < solutions; i++)
    {    
      ROS_DEBUG_STREAM("Normals " << normals_decomp[i].at<double>(normal_axis));

      if (normals_decomp[i].at<double>(normal_axis) > base && normals_decomp[i].at<double>(normal_axis) < 1.0)
      {
        index = i;
        base = normals_decomp[i].at<double>(normal_axis);
      }
    }

    if (base < normals_sublimit) 
    {
      ROS_ERROR_STREAM("Normal NOT Correctly Found " << base );
      ROS_DEBUG_STREAM("Number of Solutions: " << solutions);
      GroundTextureOdometry::matchFrame(secundum_frame, secundum_keypoints, secundum_descriptors);
      return;
    }

    // Solution Outputs
    if (debug_)
    {
      cv::Mat rvec_decomp_debug;
      for(int i = 0; i < solutions; i++)
      {
        cv::Rodrigues(Rs_decomp[i], rvec_decomp_debug);
        ROS_INFO_STREAM("Solution: " << i << " : " << index << ":");
        ROS_INFO_STREAM("rvec    : " << rvec_decomp_debug.t());
        ROS_INFO_STREAM("tvec    : " << factor_d1 * ts_decomp[i].t());
        ROS_INFO_STREAM("normal  : " << normals_decomp[i].t());
      }
    }
    
    cv::Mat tvec_decomp, rvec_decomp;
    cv::Mat tvec_decomp_temp, rvec_decomp_temp;

    tvec_decomp_temp = factor_d1 * ts_decomp[index];
    cv::Rodrigues(Rs_decomp[index], rvec_decomp_temp);
    
    tvec_decomp_temp.convertTo(tvec_decomp, CV_32F);
    rvec_decomp_temp.convertTo(rvec_decomp, CV_32F);

    cv::patchNaNs(tvec_decomp, 0.0); // An Idiotic Problem 
    cv::patchNaNs(rvec_decomp, 0.0); // An Idiotic Problem 

    if ((tvec_decomp.at<float>(0,0) < -1.0 * factor_d1|| tvec_decomp.at<float>(0,0) > 1.0*factor_d1) || (tvec_decomp.at<float>(1,0) < -1.0*factor_d1 || tvec_decomp.at<float>(1,0) > 1.0*factor_d1))
    {
      ROS_ERROR_STREAM("Out of Range " << tvec_decomp.at<float>(0,0) << " " << tvec_decomp.at<float>(1,0) << " " <<  tvec_decomp.at<float>(2,0));
      GroundTextureOdometry::matchFrame(secundum_frame, secundum_keypoints, secundum_descriptors);
      return;
    } 

    ROS_DEBUG_STREAM("XYZ " << tvec_decomp.at<float>(0,0) << " " << tvec_decomp.at<float>(1,0) << " " <<  tvec_decomp.at<float>(2,0));
    ROS_DEBUG_STREAM("RPY " << rvec_decomp.at<float>(0,0) << " " << rvec_decomp.at<float>(1,0) << " " <<  rvec_decomp.at<float>(2,0));

    y += tvec_decomp.at<float>(0,0);
    x += tvec_decomp.at<float>(1,0);
    //z += tvec_decomp.at<float>(2,0);

    //msgs.twist.twist.linear.x = tvec_decomp.at<float>(0,0);
    //msgs.twist.twist.linear.y = tvec_decomp.at<float>(1,0);
    //msgs.twist.twist.linear.z = tvec_decomp.at<float>(2,0);

    roll = 0; //+= rvec_decomp.at<float>(0,0);
    pitch = 0; //+= rvec_decomp.at<float>(1,0);
    yaw += rvec_decomp.at<float>(2,0);

    geometry_msgs::TransformStamped transform_msgs;

    transform_msgs.header.stamp = ros::Time::now();
    transform_msgs.header.frame_id = "map";
    transform_msgs.child_frame_id = "gto_odom";

    transform_msgs.transform.translation.x = x;
    transform_msgs.transform.translation.y = y;

    //msgs.twist.twist.angular.x = rvec_decomp.at<float>(0,0);
    //msgs.twist.twist.angular.y = rvec_decomp.at<float>(1,0);
    //msgs.twist.twist.angular.z = rvec_decomp.at<float>(2,0);

    //msgs.pose.pose.position.x = x;
    //msgs.pose.pose.position.y = y;
    //msgs.pose.pose.position.z = z;

    tf2::Quaternion quat;
    quat.setRPY(roll, pitch, yaw);

    transform_msgs.transform.rotation.w = quat.getW();
    transform_msgs.transform.rotation.x = quat.getX();
    transform_msgs.transform.rotation.y = quat.getY();
    transform_msgs.transform.rotation.z = quat.getZ();

    tf_br.sendTransform(transform_msgs);

    //msgs.pose.pose.orientation.w = quat.getW();
    //msgs.pose.pose.orientation.x = quat.getX();
    //msgs.pose.pose.orientation.y = quat.getY();
    //msgs.pose.pose.orientation.z = quat.getZ();

    //msgs.header.stamp = ros::Time::now();

    //output_odom_.publish(msgs);
    
    GroundTextureOdometry::matchFrame(secundum_frame, secundum_keypoints, secundum_descriptors);
  }

  void GroundTextureOdometry::publishOdom()
  {
    msgs.header.stamp = ros::Time::now();
    output_odom_.publish(msgs);
  }

  void GroundTextureOdometry::matchFrame(cv::Mat& secundum_frame, std::vector<cv::KeyPoint>& secundum_keypoints, cv::Mat& secundum_descriptors)
  {
    primum_frame = secundum_frame;
    primum_keypoints = secundum_keypoints;
    primum_descriptors = secundum_descriptors;
  }

  void GroundTextureOdometry::paramCallback(gto::gtoConfig &config, uint32_t level)
  {
  	std::lock_guard<std::mutex> lock(mutex);
    ROS_WARN("Param Update");
    
    debug_ = config.debug;

    orb->setEdgeThreshold(config.orb_edgeThreshold);
    orb->setFastThreshold(config.orb_fastTreshold);
    orb->setFirstLevel(config.orb_firstLevel);
    orb->setMaxFeatures(config.orb_nfeatures);
    orb->setNLevels(config.orb_nlevels);
    orb->setPatchSize(config.orb_pathSize);
    orb->setScaleFactor(config.orb_scaleFactor);
    orb->setScoreType(config.orb_scoreType);
    orb->setWTA_K(config.orb_WTA_K);

    matcher_type = config.matcher_type;
    matcher_knn_k = config.matcher_knn_k;
    matcher_radius = config.matcher_radius;

    nn_match_ratio = config.nn_match_ratio;

    homography_method = config.homography_method;
    homography_ransac_threshold = config.homography_ransac_threshold;
    homography_max_iter = config.homography_max_iter;
    homography_confidence = config.homography_confidence;
    factor_d1 = config.factor_d1;
    normals_sublimit = config.normals_sublimit;
    normal_axis = config.normal_axis;
  }

  void GroundTextureOdometry::initialFrame()
  {
    primum_frame = image_queue_.front();
    image_queue_.pop_front();

    orb->detectAndCompute(primum_frame, cv::noArray(), primum_keypoints, primum_descriptors);

    // Show Keypoints of Initial Frame
    cv::Mat keypoint_frame;
    cv::drawKeypoints( primum_frame, primum_keypoints, keypoint_frame, cv::Scalar( 0, 200, 0 ));
    sensor_msgs::Image ros_keypoint_image = GroundTextureOdometry::cvToRosImage(keypoint_frame);
    keypoint_publisher_.publish(ros_keypoint_image);
  }

  sensor_msgs::Image GroundTextureOdometry::cvToRosImage(const cv::Mat frame )
  {
    // Image Converter
    std_msgs::Header header; 
    header.stamp = ros::Time::now();
    cv_bridge::CvImage cv_bridge_image = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame);
    sensor_msgs::Image ros_image;
    cv_bridge_image.toImageMsg(ros_image);
    return ros_image;
  }

  void GroundTextureOdometry::imageCallback(const sensor_msgs::CompressedImage::ConstPtr& compressed_image)
  {
    // std::lock_guard<std::mutex> lock(mutex);

    cv_bridge::CvImagePtr cv_img_msg = cv_bridge::toCvCopy(compressed_image, "mono8");
    image_queue_.push_back(cv_img_msg->image);
  }


  void GroundTextureOdometry::infoCallback(const sensor_msgs::CameraInfo::ConstPtr& cam_info)
  {
    //std::lock_guard<std::mutex> lock(mutex);

    K = cam_info->K;
    D = cam_info->D;
  }

}; // namespace ground_texture_odometry
