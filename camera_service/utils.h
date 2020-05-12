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
#include "transformation.h"

class Utils {
 public:
  struct color_point_t {
    int16_t xyz[3];
    uint8_t rgb[3];
  };

  struct my_capture {
  	k4a::capture main;
  	k4a::capture secondary;
  };

  static cv::Mat color_to_opencv(const k4a::image &im);
  static cv::Mat depth_to_opencv(const k4a::image &im);
  static k4a_device_configuration_t get_default_config();
  static k4a_device_configuration_t get_master_config();
  static k4a_device_configuration_t get_subordinate_config();
  static k4a::image create_depth_image_like(const k4a::image &im);
  static void create_xy_table(const k4a_calibration_t *calibration,
                              k4a_image_t xy_table);
  static void generate_point_cloud(const k4a_image_t depth_image,
                                   const k4a_image_t xy_table,
                                   k4a_image_t point_cloud, int *point_count);
  static void write_point_cloud(const char *file_name,
                                const k4a_image_t point_cloud, int point_count);
  static void my_test(k4a_image_t depth_image, k4a_calibration_t calibration,
                      std::string file_name);
  static bool point_cloud_depth_to_color(k4a::transformation &transformation,
                                         const k4a::image &depth_image,
                                         const k4a::image &color_image,
                                         std::string file_name,
                                         const Transformation *tr = NULL);
  static bool Utils::point_cloud_custom(k4a::transformation &transformation,
                                        const k4a::image &depth_image,
                                        const k4a::image &color_image,
                                        const k4a::image &target_color_image,
                                        std::string file_name,
                                        const Transformation *tr = NULL);
  static void tranformation_helpers_write_point_cloud(
      const k4a_image_t point_cloud_image, const k4a_image_t color_image,
      const char *file_name, const Transformation *tr = NULL);
};