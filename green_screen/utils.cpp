
#include "utils.h"

#include <math.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <k4a/k4a.hpp>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
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

// Allowing at least 160 microseconds between depth cameras should ensure they
// do not interfere with one another.
constexpr unsigned int MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160;

cv::Mat Utils::color_to_opencv(const k4a::image &im) {
  cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(),
                              CV_8UC4, (void *)im.get_buffer());
  cv::Mat cv_image_no_alpha;
  cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
  return cv_image_no_alpha;
}

cv::Mat Utils::depth_to_opencv(const k4a::image &im) {
  return cv::Mat(im.get_height_pixels(), im.get_width_pixels(), CV_16U,
                 (void *)im.get_buffer(),
                 static_cast<size_t>(im.get_stride_bytes()));
}

// The following functions provide the configurations that should be used for
// each camera. NOTE: For best results both cameras should have the same
// configuration (framerate, resolution, color and depth modes). Additionally
// the both master and subordinate should have the same exposure and power line
// settings. Exposure settings can be different but the subordinate must have a
// longer exposure from master. To synchronize a master and subordinate with
// different exposures the user should set `subordinate_delay_off_master_usec =
// ((subordinate exposure time) - (master exposure time))/2`.
//
k4a_device_configuration_t Utils::get_default_config() {
  k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  camera_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  camera_config.depth_mode =
      K4A_DEPTH_MODE_WFOV_UNBINNED;  // No need for depth during calibration
  camera_config.camera_fps =
      K4A_FRAMES_PER_SECOND_15;  // Don't use all USB bandwidth
  camera_config.subordinate_delay_off_master_usec =
      0;  // Must be zero for master
  camera_config.synchronized_images_only = true;
  return camera_config;
}

// Master customizable settings
k4a_device_configuration_t Utils::get_master_config() {
  k4a_device_configuration_t camera_config = get_default_config();
  camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;

  // Two depth images should be seperated by
  // MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
  // sensor doesn't interfere with the other. To accomplish this the master
  // depth image captures (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2)
  // before the color image, and the subordinate camera captures its depth image
  // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image.
  // This gives us two depth images centered around the color image as closely
  // as possible.
  camera_config.depth_delay_off_color_usec =
      -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
  camera_config.synchronized_images_only = true;
  return camera_config;
}

// Subordinate customizable settings
k4a_device_configuration_t Utils::get_subordinate_config() {
  k4a_device_configuration_t camera_config = get_default_config();
  camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;

  // Two depth images should be seperated by
  // MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC to ensure the depth imaging
  // sensor doesn't interfere with the other. To accomplish this the master
  // depth image captures (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2)
  // before the color image, and the subordinate camera captures its depth image
  // (MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2) after the color image.
  // This gives us two depth images centered around the color image as closely
  // as possible.
  camera_config.depth_delay_off_color_usec =
      MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
  return camera_config;
}

k4a::image Utils::create_depth_image_like(const k4a::image &im) {
  return k4a::image::create(
      K4A_IMAGE_FORMAT_DEPTH16, im.get_width_pixels(), im.get_height_pixels(),
      im.get_width_pixels() * static_cast<int>(sizeof(uint16_t)));
}

void Utils::create_xy_table(const k4a_calibration_t *calibration,
                            k4a_image_t xy_table) {
  k4a_float2_t *table_data =
      (k4a_float2_t *)(void *)k4a_image_get_buffer(xy_table);

  int width = calibration->depth_camera_calibration.resolution_width;
  int height = calibration->depth_camera_calibration.resolution_height;

  k4a_float2_t p;
  k4a_float3_t ray;
  int valid;

  for (int y = 0, idx = 0; y < height; y++) {
    p.xy.y = (float)y;
    for (int x = 0; x < width; x++, idx++) {
      p.xy.x = (float)x;

      k4a_calibration_2d_to_3d(calibration, &p, 1.f, K4A_CALIBRATION_TYPE_DEPTH,
                               K4A_CALIBRATION_TYPE_DEPTH, &ray, &valid);

      if (valid) {
        table_data[idx].xy.x = ray.xyz.x;
        table_data[idx].xy.y = ray.xyz.y;
      } else {
        table_data[idx].xy.x = nanf("");
        table_data[idx].xy.y = nanf("");
      }
    }
  }
}

void Utils::generate_point_cloud(const k4a_image_t depth_image,
                                 const k4a_image_t xy_table,
                                 k4a_image_t point_cloud, int *point_count) {
  int width = k4a_image_get_width_pixels(depth_image);
  int height = k4a_image_get_height_pixels(depth_image);

  uint16_t *depth_data = (uint16_t *)(void *)k4a_image_get_buffer(depth_image);
  k4a_float2_t *xy_table_data =
      (k4a_float2_t *)(void *)k4a_image_get_buffer(xy_table);
  k4a_float3_t *point_cloud_data =
      (k4a_float3_t *)(void *)k4a_image_get_buffer(point_cloud);

  *point_count = 0;
  for (int i = 0; i < width * height; i++) {
    if (depth_data[i] != 0 && !isnan(xy_table_data[i].xy.x) &&
        !isnan(xy_table_data[i].xy.y)) {
      point_cloud_data[i].xyz.x = xy_table_data[i].xy.x * (float)depth_data[i];
      point_cloud_data[i].xyz.y = xy_table_data[i].xy.y * (float)depth_data[i];
      point_cloud_data[i].xyz.z = (float)depth_data[i];
      (*point_count)++;
    } else {
      point_cloud_data[i].xyz.x = nanf("");
      point_cloud_data[i].xyz.y = nanf("");
      point_cloud_data[i].xyz.z = nanf("");
    }
  }
}

void Utils::write_point_cloud(const char *file_name,
                              const k4a_image_t point_cloud, int point_count) {
  int width = k4a_image_get_width_pixels(point_cloud);
  int height = k4a_image_get_height_pixels(point_cloud);

  k4a_float3_t *point_cloud_data =
      (k4a_float3_t *)(void *)k4a_image_get_buffer(point_cloud);

  // save to the ply file
  std::ofstream ofs(file_name);  // text mode first
  ofs << "ply" << std::endl;
  ofs << "format ascii 1.0" << std::endl;
  ofs << "element vertex"
      << " " << point_count << std::endl;
  ofs << "property float x" << std::endl;
  ofs << "property float y" << std::endl;
  ofs << "property float z" << std::endl;
  ofs << "end_header" << std::endl;
  ofs.close();

  std::stringstream ss;
  for (int i = 0; i < width * height; i++) {
    if (isnan(point_cloud_data[i].xyz.x) || isnan(point_cloud_data[i].xyz.y) ||
        isnan(point_cloud_data[i].xyz.z)) {
      continue;
    }

    ss << (float)point_cloud_data[i].xyz.x << " "
       << (float)point_cloud_data[i].xyz.y << " "
       << (float)point_cloud_data[i].xyz.z << std::endl;
  }

  std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
  ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());
}

void Utils::my_test(k4a_image_t depth_image, k4a_calibration_t calibration,
                    std::string file_name) {
  k4a_image_t xy_table = NULL;
  k4a_image_t point_cloud = NULL;
  int point_count = 0;

  k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                   calibration.depth_camera_calibration.resolution_width,
                   calibration.depth_camera_calibration.resolution_height,
                   calibration.depth_camera_calibration.resolution_width *
                       (int)sizeof(k4a_float2_t),
                   &xy_table);

  create_xy_table(&calibration, xy_table);

  k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                   calibration.depth_camera_calibration.resolution_width,
                   calibration.depth_camera_calibration.resolution_height,
                   calibration.depth_camera_calibration.resolution_width *
                       (int)sizeof(k4a_float3_t),
                   &point_cloud);

  generate_point_cloud(depth_image, xy_table, point_cloud, &point_count);

  write_point_cloud(file_name.c_str(), point_cloud, point_count);

  k4a_image_release(xy_table);
  k4a_image_release(point_cloud);
}

bool Utils::point_cloud_depth_to_color(k4a::transformation &transformation,
                                       const k4a::image &depth_image,
                                       const k4a::image &color_image,
                                       std::string file_name,
                                       const Transformation *tr) {
  // transform depth image into color camera geometry
  int color_image_width_pixels =
      k4a_image_get_width_pixels(color_image.handle());
  int color_image_height_pixels =
      k4a_image_get_height_pixels(color_image.handle());
  k4a::image transformed_depth_image =
      k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16, color_image_width_pixels,
                         color_image_height_pixels,
                         color_image_width_pixels * (int)sizeof(uint16_t));

  k4a::image point_cloud_image =
      k4a::image::create(K4A_IMAGE_FORMAT_CUSTOM, color_image_width_pixels,
                         color_image_height_pixels,
                         color_image_width_pixels * 3 * (int)sizeof(int16_t));

  transformation.depth_image_to_color_camera(depth_image,
                                             &transformed_depth_image);

  transformation.depth_image_to_point_cloud(
      transformed_depth_image, K4A_CALIBRATION_TYPE_COLOR, &point_cloud_image);

  tranformation_helpers_write_point_cloud(
      point_cloud_image.handle(), color_image.handle(), file_name.c_str(), tr);

  transformed_depth_image.reset();
  point_cloud_image.reset();

  return true;
}

bool Utils::point_cloud_custom(k4a::transformation &transformation,
                               const k4a::image &depth_image,
                               const k4a::image &color_image,
                               const k4a::image &target_color_image,
                               std::string file_name, const Transformation *tr) {
  // transform depth image into color camera geometry
  int color_image_width_pixels =
      k4a_image_get_width_pixels(target_color_image.handle());
  int color_image_height_pixels =
      k4a_image_get_height_pixels(target_color_image.handle());
  k4a::image transformed_depth_image =
      k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16, color_image_width_pixels,
                         color_image_height_pixels,
                         color_image_width_pixels * (int)sizeof(uint16_t));

  k4a::image transformed_color_image =
      k4a::image::create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                         color_image_width_pixels, color_image_height_pixels,
                         color_image_width_pixels * (int)sizeof(uint32_t));

  std::cout << depth_image.get_width_pixels() << ", "
            << transformed_depth_image.get_width_pixels() << std::endl;
  std::cout << depth_image.get_height_pixels() << ", "
            << transformed_depth_image.get_height_pixels() << std::endl;
  std::cout << depth_image.get_stride_bytes() << ", "
            << transformed_depth_image.get_stride_bytes() << std::endl;

  std::cout << color_image.get_width_pixels() << ", "
            << transformed_color_image.get_width_pixels() << std::endl;
  std::cout << color_image.get_height_pixels() << ", "
            << transformed_color_image.get_height_pixels() << std::endl;
  std::cout << color_image.get_stride_bytes() << ", "
            << transformed_color_image.get_stride_bytes() << std::endl;

  std::cout << "before custom depth to color tr" << std::endl;
  transformation.depth_image_to_color_camera_custom(
      depth_image, color_image, &transformed_depth_image,
      &transformed_color_image, K4A_TRANSFORMATION_INTERPOLATION_TYPE_NEAREST,
      0);

  k4a::image point_cloud_image =
      k4a::image::create(K4A_IMAGE_FORMAT_CUSTOM, color_image_width_pixels,
                         color_image_height_pixels,
                         color_image_width_pixels * 3 * (int)sizeof(int16_t));

  std::cout << "before depth to point cloud tr" << std::endl;
  transformation.depth_image_to_point_cloud(
      transformed_depth_image, K4A_CALIBRATION_TYPE_COLOR, &point_cloud_image);

  tranformation_helpers_write_point_cloud(point_cloud_image.handle(),
                                          transformed_color_image.handle(),
                                          file_name.c_str(), tr);

  transformed_depth_image.reset();
  point_cloud_image.reset();

  return true;
}

void Utils::tranformation_helpers_write_point_cloud(
    const k4a_image_t point_cloud_image, const k4a_image_t color_image,
    const char *file_name, const Transformation *tr) {
  if (tr != NULL)
    std::cout << "using calibration for final transformation" << std::endl;

  std::vector<color_point_t> points;

  int width = k4a_image_get_width_pixels(point_cloud_image);
  int height = k4a_image_get_height_pixels(color_image);

  int16_t *point_cloud_image_data =
      (int16_t *)(void *)k4a_image_get_buffer(point_cloud_image);
  uint8_t *color_image_data = k4a_image_get_buffer(color_image);

  for (int i = 0; i < width * height; i++) {
    color_point_t point;

    if (tr == NULL) {
      point.xyz[0] = point_cloud_image_data[3 * i + 0];
      point.xyz[1] = point_cloud_image_data[3 * i + 1];
      point.xyz[2] = point_cloud_image_data[3 * i + 2];
    } else {
      k4a_float3_t source;
      source.xyz.x = point_cloud_image_data[3 * i + 0];
      source.xyz.y = point_cloud_image_data[3 * i + 1];
      source.xyz.z = point_cloud_image_data[3 * i + 2];
      k4a_float3_t target;
      target.xyz.x = tr->R(0, 0) * source.xyz.x + tr->R(0, 1) * source.xyz.y +
                     tr->R(0, 2) * source.xyz.z + tr->t[0];
      target.xyz.y = tr->R(1, 0) * source.xyz.x + tr->R(1, 1) * source.xyz.y +
                     tr->R(1, 2) * source.xyz.z + tr->t[1];
      target.xyz.z = tr->R(2, 0) * source.xyz.x + tr->R(2, 1) * source.xyz.y +
                     tr->R(2, 2) * source.xyz.z + tr->t[2];
      // if (source.xyz.x != target.xyz.x || source.xyz.y != target.xyz.y ||
      //     source.xyz.z != target.xyz.z) {
      //   std::cout << "not equal source and target" << std::endl;
      // }
      point.xyz[0] = (uint16_t)target.xyz.x;
      point.xyz[1] = (uint16_t)target.xyz.y;
      point.xyz[2] = (uint16_t)target.xyz.z;
    }

    if (point.xyz[2] == 0) {
      continue;
    }

    point.rgb[0] = color_image_data[4 * i + 0];
    point.rgb[1] = color_image_data[4 * i + 1];
    point.rgb[2] = color_image_data[4 * i + 2];
    uint8_t alpha = color_image_data[4 * i + 3];

    if (point.rgb[0] == 0 && point.rgb[1] == 0 && point.rgb[2] == 0 &&
        alpha == 0) {
      continue;
    }

    points.push_back(point);
  }

  // save to the ply file
  std::ofstream ofs(file_name);  // text mode first
  ofs << "ply" << std::endl;
  ofs << "format ascii 1.0" << std::endl;
  ofs << "element vertex"
      << " " << points.size() << std::endl;
  ofs << "property float x" << std::endl;
  ofs << "property float y" << std::endl;
  ofs << "property float z" << std::endl;
  ofs << "property uchar red" << std::endl;
  ofs << "property uchar green" << std::endl;
  ofs << "property uchar blue" << std::endl;
  ofs << "end_header" << std::endl;
  ofs.close();

  std::stringstream ss;
  for (size_t i = 0; i < points.size(); ++i) {
    // image data is BGR
    ss << (float)points[i].xyz[0] << " " << (float)points[i].xyz[1] << " "
       << (float)points[i].xyz[2];
    ss << " " << (float)points[i].rgb[2] << " " << (float)points[i].rgb[1]
       << " " << (float)points[i].rgb[0];
    ss << std::endl;
  }
  std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
  ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());
  ofs_text.close();
}