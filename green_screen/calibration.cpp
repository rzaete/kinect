#include "calibration.h"

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
#include "utils.h"

cv::Matx33f Calibration::calibration_to_color_camera_matrix(
    const k4a::calibration &cal) {
  const k4a_calibration_intrinsic_parameters_t::_param &i =
      cal.color_camera_calibration.intrinsics.parameters.param;
  cv::Matx33f camera_matrix = cv::Matx33f::eye();
  camera_matrix(0, 0) = i.fx;
  camera_matrix(1, 1) = i.fy;
  camera_matrix(0, 2) = i.cx;
  camera_matrix(1, 2) = i.cy;
  return camera_matrix;
}

Transformation Calibration::get_depth_to_color_transformation_from_calibration(
    const k4a::calibration &cal) {
  const k4a_calibration_extrinsics_t &ex =
      cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
  Transformation tr;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      tr.R(i, j) = ex.rotation[i * 3 + j];
    }
  }
  tr.t = cv::Vec3d(ex.translation[0], ex.translation[1], ex.translation[2]);
  return tr;
}

// This function constructs a calibration that operates as a transformation
// between the secondary device's depth camera and the main camera's color
// camera. IT WILL NOT GENERALIZE TO OTHER TRANSFORMS. Under the hood, the
// transformation depth_image_to_color_camera method can be thought of as
// converting each depth pixel to a 3d point using the intrinsics of the depth
// camera, then using the calibration's extrinsics to convert between depth and
// color, then using the color intrinsics to produce the point in the color
// camera perspective.
k4a::calibration Calibration::construct_device_to_device_calibration(
    const k4a::calibration &main_cal, const k4a::calibration &secondary_cal,
    const Transformation &secondary_to_main) {
  k4a::calibration cal = secondary_cal;
  k4a_calibration_extrinsics_t &ex =
      cal.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ex.rotation[i * 3 + j] = static_cast<float>(secondary_to_main.R(i, j));
    }
  }
  for (int i = 0; i < 3; ++i) {
    ex.translation[i] = static_cast<float>(secondary_to_main.t[i]);
  }
  cal.color_camera_calibration = main_cal.color_camera_calibration;
  return cal;
}

k4a::calibration Calibration::construct_device_to_device_color_calibration(
    const k4a::calibration &main_cal, const k4a::calibration &secondary_cal,
    const Transformation &secondary_to_main) {
  k4a::calibration cal = secondary_cal;
  k4a_calibration_extrinsics_t &ex =
      cal.extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_COLOR];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ex.rotation[i * 3 + j] = static_cast<float>(secondary_to_main.R(i, j));
    }
  }
  for (int i = 0; i < 3; ++i) {
    ex.translation[i] = static_cast<float>(secondary_to_main.t[i]);
  }
  cal.color_camera_calibration = main_cal.color_camera_calibration;
  return cal;
}

vector<float> Calibration::calibration_to_color_camera_dist_coeffs(
    const k4a::calibration &cal) {
  const k4a_calibration_intrinsic_parameters_t::_param &i =
      cal.color_camera_calibration.intrinsics.parameters.param;
  return {i.k1, i.k2, i.p1, i.p2, i.k3, i.k4, i.k5, i.k6};
}

bool Calibration::find_chessboard_corners_helper(
    const cv::Mat &main_color_image, const cv::Mat &secondary_color_image,
    const cv::Size &chessboard_pattern,
    vector<cv::Point2f> &main_chessboard_corners,
    vector<cv::Point2f> &secondary_chessboard_corners) {
  std::cout << "before cv funcs" << std::endl;
  bool found_chessboard_main = cv::findChessboardCorners(
      main_color_image, chessboard_pattern, main_chessboard_corners);
  bool found_chessboard_secondary = cv::findChessboardCorners(
      secondary_color_image, chessboard_pattern, secondary_chessboard_corners);

  std::cout << "after cv funcs" << std::endl;
  // Cover the failure cases where chessboards were not found in one or both
  // images.
  if (!found_chessboard_main || !found_chessboard_secondary) {
    if (found_chessboard_main) {
      cout << "Could not find the chessboard corners in the secondary image. "
              "Trying again...\n";
    }
    // Likewise, if the chessboard was found in the secondary image, it was not
    // found in the main image.
    else if (found_chessboard_secondary) {
      cout << "Could not find the chessboard corners in the main image. Trying "
              "again...\n";
    }
    // The only remaining case is the corners were in neither image.
    else {
      cout << "Could not find the chessboard corners in either image. Trying "
              "again...\n";
    }
    return false;
  }
  // Before we go on, there's a quick problem with calibration to address.
  // Because the chessboard looks the same when rotated 180 degrees, it is
  // possible that the chessboard corner finder may find the correct points, but
  // in the wrong order.

  // A visual:
  //        Image 1                  Image 2
  // .....................    .....................
  // .....................    .....................
  // .........xxxxx2......    .....xxxxx1..........
  // .........xxxxxx......    .....xxxxxx..........
  // .........xxxxxx......    .....xxxxxx..........
  // .........1xxxxx......    .....2xxxxx..........
  // .....................    .....................
  // .....................    .....................

  // The problem occurs when this case happens: the find_chessboard() function
  // correctly identifies the points on the chessboard (shown as 'x's) but the
  // order of those points differs between images taken by the two cameras.
  // Specifically, the first point in the list of points found for the first
  // image (1) is the *last* point in the list of points found for the second
  // image (2), though they correspond to the same physical point on the
  // chessboard.

  // To avoid this problem, we can make the assumption that both of the cameras
  // will be oriented in a similar manner (e.g. turning one of the cameras
  // upside down will break this assumption) and enforce that the vector between
  // the first and last points found in pixel space (which will be at opposite
  // ends of the chessboard) are pointing the same direction- so, the dot
  // product of the two vectors is positive.

  cv::Vec2f main_image_corners_vec =
      main_chessboard_corners.back() - main_chessboard_corners.front();
  cv::Vec2f secondary_image_corners_vec = secondary_chessboard_corners.back() -
                                          secondary_chessboard_corners.front();
  if (main_image_corners_vec.dot(secondary_image_corners_vec) <= 0.0) {
    std::reverse(secondary_chessboard_corners.begin(),
                 secondary_chessboard_corners.end());
  }
  return true;
}

Transformation Calibration::stereo_calibration(
    const k4a::calibration &main_calib, const k4a::calibration &secondary_calib,
    const vector<vector<cv::Point2f>> &main_chessboard_corners_list,
    const vector<vector<cv::Point2f>> &secondary_chessboard_corners_list,
    const cv::Size &image_size, const cv::Size &chessboard_pattern,
    float chessboard_square_length) {
  // We have points in each image that correspond to the corners that the
  // findChessboardCorners function found. However, we still need the points in
  // 3 dimensions that these points correspond to. Because we are ultimately
  // only interested in find a transformation between two cameras, these points
  // don't have to correspond to an external "origin" point. The only important
  // thing is that the relative distances between points are accurate. As a
  // result, we can simply make the first corresponding point (0, 0) and
  // construct the remaining points based on that one. The order of points
  // inserted into the vector here matches the ordering of
  // findChessboardCorners. The units of these points are in millimeters, mostly
  // because the depth provided by the depth cameras is also provided in
  // millimeters, which makes for easy comparison.
  vector<cv::Point3f> chessboard_corners_world;
  for (int h = 0; h < chessboard_pattern.height; ++h) {
    for (int w = 0; w < chessboard_pattern.width; ++w) {
      chessboard_corners_world.emplace_back(cv::Point3f{
          w * chessboard_square_length, h * chessboard_square_length, 0.0});
    }
  }

  // Calibrating the cameras requires a lot of data. OpenCV's stereoCalibrate
  // function requires:
  // - a list of points in real 3d space that will be used to calibrate*
  // - a corresponding list of pixel coordinates as seen by the first camera*
  // - a corresponding list of pixel coordinates as seen by the second camera*
  // - the camera matrix of the first camera
  // - the distortion coefficients of the first camera
  // - the camera matrix of the second camera
  // - the distortion coefficients of the second camera
  // - the size (in pixels) of the images
  // - R: stereoCalibrate stores the rotation matrix from the first camera to
  // the second here
  // - t: stereoCalibrate stores the translation vector from the first camera to
  // the second here
  // - E: stereoCalibrate stores the essential matrix here (we don't use this)
  // - F: stereoCalibrate stores the fundamental matrix here (we don't use this)
  //
  // * note: OpenCV's stereoCalibrate actually requires as input an array of
  // arrays of points for these arguments, allowing a caller to provide multiple
  // frames from the same camera with corresponding points. For example, if
  // extremely high precision was required, many images could be taken with each
  // camera, and findChessboardCorners applied to each of those images, and
  // OpenCV can jointly solve for all of the pairs of corresponding images.
  // However, to keep things simple, we use only one image from each device to
  // calibrate.  This is also why each of the vectors of corners is placed into
  // another vector.
  //
  // A function in OpenCV's calibration function also requires that these points
  // be F32 types, so we use those. However, OpenCV still provides doubles as
  // output, strangely enough.
  vector<vector<cv::Point3f>> chessboard_corners_world_nested_for_cv(
      main_chessboard_corners_list.size(), chessboard_corners_world);

  cv::Matx33f main_camera_matrix =
      calibration_to_color_camera_matrix(main_calib);
  cv::Matx33f secondary_camera_matrix =
      calibration_to_color_camera_matrix(secondary_calib);
  vector<float> main_dist_coeff =
      calibration_to_color_camera_dist_coeffs(main_calib);
  vector<float> secondary_dist_coeff =
      calibration_to_color_camera_dist_coeffs(secondary_calib);

  // Finally, we'll actually calibrate the cameras.
  // Pass secondary first, then main, because we want a transform from secondary
  // to main.
  Transformation tr;
  double error = cv::stereoCalibrate(
      chessboard_corners_world_nested_for_cv, secondary_chessboard_corners_list,
      main_chessboard_corners_list, secondary_camera_matrix,
      secondary_dist_coeff, main_camera_matrix, main_dist_coeff, image_size,
      tr.R,  // output
      tr.t,  // output
      cv::noArray(), cv::noArray(),
      cv::CALIB_FIX_INTRINSIC | cv::CALIB_RATIONAL_MODEL |
          cv::CALIB_CB_FAST_CHECK);
  cout << "Finished calibrating!\n";
  cout << "Got error of " << error << "\n";
  return tr;
}

Transformation Calibration::calibrate_devices(
    MultiDeviceCapturer &capturer,
    const k4a_device_configuration_t &main_config,
    const k4a_device_configuration_t &secondary_config,
    const cv::Size &chessboard_pattern, float chessboard_square_length,
    double calibration_timeout) {
  k4a::calibration main_calibration =
      capturer.get_master_device().get_calibration(
          main_config.depth_mode, main_config.color_resolution);

  k4a::calibration secondary_calibration =
      capturer.get_subordinate_device_by_index(0).get_calibration(
          secondary_config.depth_mode, secondary_config.color_resolution);
  vector<vector<cv::Point2f>> main_chessboard_corners_list;
  vector<vector<cv::Point2f>> secondary_chessboard_corners_list;
  std::chrono::time_point<std::chrono::system_clock> start_time =
      std::chrono::system_clock::now();
  while (std::chrono::duration<double>(std::chrono::system_clock::now() -
                                       start_time)
             .count() < calibration_timeout) {
    vector<k4a::capture> captures =
        capturer.get_synchronized_captures(secondary_config);
    std::cout << "new calibration capture" << std::endl;
    k4a::capture &main_capture = captures[0];
    k4a::capture &secondary_capture = captures[1];
    // get_color_image is guaranteed to be non-null because we use
    // get_synchronized_captures for color (get_synchronized_captures also
    // offers a flag to use depth for the secondary camera instead of color).
    k4a::image main_color_image = main_capture.get_color_image();
    k4a::image secondary_color_image = secondary_capture.get_color_image();
    cv::Mat cv_main_color_image = Utils::color_to_opencv(main_color_image);
    cv::Mat cv_secondary_color_image =
        Utils::color_to_opencv(secondary_color_image);

    vector<cv::Point2f> main_chessboard_corners;
    vector<cv::Point2f> secondary_chessboard_corners;
    std::cout << "start find corners" << std::endl;
    bool got_corners = Calibration::find_chessboard_corners_helper(
        cv_main_color_image, cv_secondary_color_image, chessboard_pattern,
        main_chessboard_corners, secondary_chessboard_corners);
    std::cout << "finish find corners" << std::endl;
    if (got_corners) {
      main_chessboard_corners_list.emplace_back(main_chessboard_corners);
      secondary_chessboard_corners_list.emplace_back(
          secondary_chessboard_corners);
      cv::drawChessboardCorners(cv_main_color_image, chessboard_pattern,
                                main_chessboard_corners, true);
      cv::drawChessboardCorners(cv_secondary_color_image, chessboard_pattern,
                                secondary_chessboard_corners, true);
    }

    cv::imshow("Chessboard view from main camera", cv_main_color_image);
    cv::waitKey(1);
    cv::imshow("Chessboard view from secondary camera",
               cv_secondary_color_image);
    cv::waitKey(1);

    // Get 20 frames before doing calibration.
    if (main_chessboard_corners_list.size() >= 5) {
      cout << "Calculating calibration..." << endl;
      return stereo_calibration(
          main_calibration, secondary_calibration, main_chessboard_corners_list,
          secondary_chessboard_corners_list, cv_main_color_image.size(),
          chessboard_pattern, chessboard_square_length);
    }
  }
  std::cerr << "Calibration timed out !\n ";
  exit(1);
}
