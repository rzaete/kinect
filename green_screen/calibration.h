#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <k4a/k4a.hpp>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

#include "MultiDeviceCapturer.h"
#include "calibration.h"
#include "transformation.h"
#include "utils.h"

class Calibration {
 public:
  static cv::Matx33f calibration_to_color_camera_matrix(
      const k4a::calibration &cal);
  static Transformation get_depth_to_color_transformation_from_calibration(
      const k4a::calibration &cal);
  static k4a::calibration construct_device_to_device_calibration(
      const k4a::calibration &main_cal, const k4a::calibration &secondary_cal,
      const Transformation &secondary_to_main);
    static k4a::calibration construct_device_to_device_color_calibration(
      const k4a::calibration &main_cal, const k4a::calibration &secondary_cal,
      const Transformation &secondary_to_main);
  static vector<float> calibration_to_color_camera_dist_coeffs(
      const k4a::calibration &cal);
  static bool find_chessboard_corners_helper(
      const cv::Mat &main_color_image, const cv::Mat &secondary_color_image,
      const cv::Size &chessboard_pattern,
      vector<cv::Point2f> &main_chessboard_corners,
      vector<cv::Point2f> &secondary_chessboard_corners);
  static Transformation stereo_calibration(
      const k4a::calibration &main_calib,
      const k4a::calibration &secondary_calib,
      const vector<vector<cv::Point2f>> &main_chessboard_corners_list,
      const vector<vector<cv::Point2f>> &secondary_chessboard_corners_list,
      const cv::Size &image_size, const cv::Size &chessboard_pattern,
      float chessboard_square_length);

  static Transformation calibrate_devices(
      MultiDeviceCapturer &capturer,
      const k4a_device_configuration_t &main_config,
      const k4a_device_configuration_t &secondary_config,
      const cv::Size &chessboard_pattern, float chessboard_square_length,
      double calibration_timeout);
};