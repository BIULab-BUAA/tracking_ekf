/*
 * @Author: xindong324
 * @Date: 2022-05-22 19:21:41
 * @LastEditors: xindong324
 * @LastEditTime: 2022-05-25 23:05:09
 * @Description: file content
 */
#include<ros/ros.h>
#include <Eigen/Geometry>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <aruco/fractaldetector.h>
#include <aruco/aruco_cvversioning.h>
#include <aruco/cvdrawingutils.h>
#include <aruco/aruco.h>
#include <aruco/cameraparameters.h>
#include "target_detection/target_ekf.hpp"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
        ImageOdomSyncPolicy;
typedef message_filters::Synchronizer<ImageOdomSyncPolicy>
        ImageOdomSynchronizer;



struct CamConfig{
    double rate;
    double range;
    int width;
    int height;
    double fx;
    double fy;
    double cx;
    double cy;
};


std::shared_ptr<Ekf> ekfPtr_;

class ArucoDetector{

private:  
    // aruco 
    double markerSize_;
    aruco::FractalDetector fDetector_;

    // cam config
    CamConfig camConfig_;
    Eigen::Matrix3d cam2body_R_, base2mark_R_;
    Eigen::Vector3d cam2body_p_, base2mark_p_;

    std::shared_ptr<Ekf> ekfPtr_;

    // 需要设置为类成员或者全局变量
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    std::shared_ptr<ImageOdomSynchronizer> image_odom_sync_Ptr_;

    // topic sub/pub var
    nav_msgs::Odometry uav_odom, aruco_base_odom;
    // 需要设置为类成员或者全局变量
    ros::Timer ekf_predict_timer_;

    // sub and pub
    
    ros::Publisher target_odom_pub_, yolo_odom_pub;
    

    void image_odom_callback(const sensor_msgs::ImageConstPtr& image_msg,
                             const nav_msgs::OdometryConstPtr& odom_msg){
                         
        Eigen::Vector3d uav_p(odom_msg->pose.pose.position.x,
                               odom_msg->pose.pose.position.y,
                               odom_msg->pose.pose.position.z);
        Eigen::Quaterniond uav_q(
            odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);

        Eigen::Vector3d cam_p = uav_q.toRotationMatrix() * cam2body_p_ + uav_p;
        Eigen::Quaterniond cam_q = uav_q * Eigen::Quaterniond(cam2body_R_);

        // detect marker2cam_R, marker2cam_q
        cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(image_msg);
        
        cv::Mat image = image_ptr->image;
        //ROS_INFO("img_size:w: %d, h: %d",image.rows, image.cols );
        if(!fDetector_.detect(image)){
            //ROS_INFO("NO MARKER");
            return;
        }

        if(fDetector_.poseEstimation()){

            cv::Mat tvec = fDetector_.getTvec();
            cv::Mat rvec = fDetector_.getRvec();
            cv::Mat rotM,rotT;
            cv::Rodrigues(rvec,rotM);
            
            Eigen::Matrix3d marker2cam_R;
            Eigen::Vector3d marker2cam_p( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
            marker2cam_R<<rotM.at<double>(0,0), rotM.at<double>(0,1), rotM.at<double>(0,2),
                        rotM.at<double>(1,0), rotM.at<double>(1,1), rotM.at<double>(1,2),
                        rotM.at<double>(2,0), rotM.at<double>(2,1), rotM.at<double>(2,2);
            // std::cout << "marker2cam_R: "<< marker2cam_R << std::endl;

            // transport marker_p
            Eigen::Vector3d marker_p = cam_q.toRotationMatrix() * marker2cam_p + cam_p;
            Eigen::Quaterniond marker_q = cam_q * Eigen::Quaterniond(marker2cam_R);
            // std::cout << "markerp: "<< marker_p << std::endl;
            // std::cout << "markerq: "<< marker_q.toRotationMatrix() << std::endl;
            //transport link_p using link2marker_p, link2marker_R  
            Eigen::Vector3d link_p = marker_q.toRotationMatrix() * base2mark_p_ + marker_p;
            Eigen::Quaterniond link_q = marker_q * Eigen::Quaterniond(base2mark_R_);
            // std::cout << "linkp: "<< link_p<< std::endl;
            // std::cout << "linkq: "<< link_q.toRotationMatrix()<< std::endl;
            // std::cout << "marker r2" << base2mark_R_ <<  std::endl;
            // std::cout << "marker_R" << Eigen::Quaterniond(base2mark_R_).toRotationMatrix()<< std::endl;
             // ekf update
            // update target odom
            Eigen::Vector3d rpy = quaternion2euler(link_q);
            // ROS_INFO("link x:%f, y: %f, z:%f, r: %f, p: %f, y: %f", 
            // link_p.x(), link_p.y(), link_p.z(), rpy.x(), rpy.y(), rpy.z());

            double update_dt = (ros::Time::now() - last_update_stamp_).toSec();
             if (update_dt > 5.0) {
                ekfPtr_->reset(link_p, rpy);
                ROS_WARN("[ekf] reset!");
            } else if (ekfPtr_->update(link_p, rpy)) {
                // ROS_WARN("[ekf] update!");
            } else {
                ROS_ERROR("[ekf] update invalid!");
                return;
            }
            last_update_stamp_ = ros::Time::now();
        }    
        //pub                  
    }

    void predict_state_callback(const ros::TimerEvent& event){
        double update_dt = (ros::Time::now() - last_update_stamp_).toSec();
        if(update_dt < 2.0){
            ekfPtr_->predict();
        }else{
            ROS_WARN("[ekf] too long time no update");
        }

        // ROS_INFO("odom x:%f, y: %f, z:%f, r: %f, p: %f, y: %f", 
        //     ekfPtr_->pos().x(), ekfPtr_->pos().y(), ekfPtr_->pos().z(), ekfPtr_->rpy().x(), ekfPtr_->rpy().y(), ekfPtr_->rpy().z());

        // publish target odom
        nav_msgs::Odometry target_odom;
        target_odom.header.stamp = ros::Time::now();
        target_odom.header.frame_id = "world";
        target_odom.pose.pose.position.x = ekfPtr_->pos().x();
        target_odom.pose.pose.position.y = ekfPtr_->pos().y();
        target_odom.pose.pose.position.z = ekfPtr_->pos().z();
        target_odom.twist.twist.linear.x = ekfPtr_->vel().x();
        target_odom.twist.twist.linear.y = ekfPtr_->vel().y();
        target_odom.twist.twist.linear.z = ekfPtr_->vel().z();
        Eigen::Vector3d rpy = ekfPtr_->rpy();
        Eigen::Quaterniond q = euler2quaternion(rpy);
        target_odom.pose.pose.orientation.w = q.w();
        target_odom.pose.pose.orientation.x = q.x();
        target_odom.pose.pose.orientation.y = q.y();
        target_odom.pose.pose.orientation.z = q.z();
        target_odom_pub_.publish(target_odom);
    }

public:
    ros::Time last_update_stamp_;
    void init(ros::NodeHandle& nh){
        // get param 
        std::vector<double> tmp;
        if(nh.param<std::vector<double>>("cam2body_R", tmp, std::vector<double>())){
            cam2body_R_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 3);
        }
        if(nh.param<std::vector<double>>("cam2body_p", tmp, std::vector<double>())){
            cam2body_p_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 1);
        }

        // base 2 aruco trans
        if(nh.param<std::vector<double>>("body2aruco_R", tmp, std::vector<double>())){
            base2mark_R_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 3);
        }
        if(nh.param<std::vector<double>>("body2aruco_p", tmp, std::vector<double>())){
            base2mark_p_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 1);
        }


        // aruco param config
        nh.getParam("marker_size", markerSize_);
        ROS_INFO("aruco marker_size: %f", markerSize_);
        std::string aruco_config_path, aruco_type;

        nh.param<std::string>("aruco_config_path", aruco_config_path, "");
        nh.param<std::string>("aruco_type", aruco_type, "");

        
        aruco::CameraParameters camParam;
        camParam.readFromXMLFile(aruco_config_path);
        fDetector_.setConfiguration(aruco_type);
        if(camParam.isValid()){
            fDetector_.setParams(camParam, markerSize_);
        }else
            ROS_INFO("cam param invalid");
        
        target_odom_pub_ = nh.advertise<nav_msgs::Odometry>("target_odom", 1);

        // ekf config
        //更新频率有点慢
        int ekf_rate = 20;
        nh.getParam("ekf_rate", ekf_rate);
        ROS_INFO("EKF_RATE: %d", ekf_rate);

        ekfPtr_ = std::make_shared<Ekf>(1.0 / ekf_rate);
        
        // sync cofig
        
        image_sub_.subscribe(nh, "image", 1, ros::TransportHints().tcpNoDelay());
        odom_sub_.subscribe(nh, "odom", 100, ros::TransportHints().tcpNoDelay());
        image_odom_sync_Ptr_ = std::make_shared<ImageOdomSynchronizer>(ImageOdomSyncPolicy(100), image_sub_, odom_sub_);
        image_odom_sync_Ptr_->registerCallback(boost::bind(&ArucoDetector::image_odom_callback, this,_1, _2));
        ekf_predict_timer_ = nh.createTimer(ros::Duration(1.0/ekf_rate), &ArucoDetector::predict_state_callback, this);
        //yolo_odom_sync_Ptr = 
        ROS_INFO("init end");
    }


      
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "aruco_detect");
    
    ros::NodeHandle nh("~");

    //ros::Duration(11.0).sleep();
    ROS_INFO("nh");
    ArucoDetector aruco_detector;
    aruco_detector.last_update_stamp_ = ros::Time::now();
    aruco_detector.init(nh);

    ros::Duration(1.0).sleep();

    ros::spin();

    return 0;


}