// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <signal.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <k4a/k4a.hpp>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
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

MultiDeviceCapturer* capturer_ptr = NULL;
bool exit_flag = false;

void sigintHandler(int sig_num) {
  exit_flag = true;
  // exit(sig_num);
}

int main(int argc, char** argv) {
  signal(SIGINT, sigintHandler);
  float chessboard_square_length = 0.;  // must be included in the input params
  int color_exposure_usec = 25000;  // somewhat reasonable default exposure time
  int powerline_freq = 2;           // default to a 60 Hz powerline
  cv::Size chessboard_pattern(0, 0);    // height, width. Both need to be set.
  unsigned int depth_threshold = 1000;  // default to 1 meter
  size_t num_devices = 0;
  double calibration_timeout =
      180.0;  // default to timing out after 60s of trying to get calibrated
  double greenscreen_duration =
      std::numeric_limits<double>::max();  // run forever
  bool enable_visualization = false;

  vector<unsigned int> device_indices{
      0};  // Set up a MultiDeviceCapturer to handle getting many synchronous
           // captures Note that the order of indices in device_indices is not
           // necessarily preserved because MultiDeviceCapturer tries to find
           // the master device based on which one has sync out plugged in.
           // Start with just { 0 }, and add another if needed

  if (argc < 5) {
    cout << "Usage: green_screen <num-cameras> <board-height> <board-width> "
            "<board-square-length> "
            "[depth-threshold-mm (default 1000)] [color-exposure-time-usec "
            "(default 8000)] "
            "[powerline-frequency-mode (default 2 for 60 Hz)] "
            "[calibration-timeout-sec (default 60)]"
            "[greenscreen-duration-sec (default infinity- run forever)]"
         << endl;

    cerr << "Not enough arguments!\n";
    exit(1);
  } else {
    num_devices = static_cast<size_t>(atoi(argv[1]));
    if (num_devices > k4a::device::get_installed_count()) {
      cerr << "Not enough cameras plugged in!\n";
      exit(1);
    }
    chessboard_pattern.height = atoi(argv[2]);
    chessboard_pattern.width = atoi(argv[3]);
    chessboard_square_length = static_cast<float>(atof(argv[4]));

    if (argc > 5) {
      depth_threshold = static_cast<uint16_t>(atoi(argv[5]));
      if (argc > 6) {
        color_exposure_usec = atoi(argv[6]);
        if (argc > 7) {
          powerline_freq = atoi(argv[7]);
          if (argc > 8) {
            calibration_timeout = atof(argv[8]);
            if (argc > 9) {
              greenscreen_duration = atof(argv[9]);
            }
          }
        }
      }
    }
  }

  if (num_devices != 2 && num_devices != 1) {
    cerr << "Invalid choice for number of devices!\n";
    exit(1);
  } else if (num_devices == 2) {
    device_indices.emplace_back(1);  // now device indices are { 0, 1 }
  }
  if (chessboard_pattern.height == 0) {
    cerr << "Chessboard height is not properly set!\n";
    exit(1);
  }
  if (chessboard_pattern.width == 0) {
    cerr << "Chessboard height is not properly set!\n";
    exit(1);
  }
  if (chessboard_square_length == 0.) {
    cerr << "Chessboard square size is not properly set!\n";
    exit(1);
  }

  cout << "Chessboard height: " << chessboard_pattern.height
       << ". Chessboard width: " << chessboard_pattern.width
       << ". Chessboard square length: " << chessboard_square_length << endl;
  cout << "Depth threshold: : " << depth_threshold
       << ". Color exposure time: " << color_exposure_usec
       << ". Powerline frequency mode: " << powerline_freq << endl;

  cv::ocl::setUseOpenCL(false);
  MultiDeviceCapturer capturer(device_indices, color_exposure_usec,
                               powerline_freq);
  capturer_ptr = &capturer;

  // Create configurations for devices
  k4a_device_configuration_t main_config = Utils::get_master_config();
  if (num_devices == 1)  // no need to have a master cable if it's standalone
  {
    main_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
  }
  k4a_device_configuration_t secondary_config = Utils::get_subordinate_config();

  // Construct all the things that we'll need whether or not we are running with
  // 1 or 2 cameras
  k4a::calibration main_calibration =
      capturer.get_master_device().get_calibration(
          main_config.depth_mode, main_config.color_resolution);

  // Set up a transformation. DO THIS OUTSIDE OF YOUR MAIN LOOP! Constructing
  // transformations involves time-intensive hardware setup and should not
  // change once you have a rigid setup, so only call it once or it will run
  // very slowly.
  k4a::transformation main_depth_to_main_color(main_calibration);

  capturer.start_devices(main_config, secondary_config);
  std::cout << "devices started" << std::endl;

  // get an image to be the background
  vector<k4a::capture> sample_captures =
      capturer.get_synchronized_captures(secondary_config);
  k4a::image sample_image = sample_captures[0].get_color_image();
  cv::Mat black_background =
      cv::Mat::zeros(sample_image.get_height_pixels(),
                     sample_image.get_width_pixels(), CV_8UC3);
  cv::Mat output_image =
      black_background.clone();  // allocated outside the loop to avoid
                                 // re-creating every time
  // cv::namedWindow("Color", cv::WINDOW_AUTOSIZE);
  // cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
  // cv::VideoWriter video("outcpp.avi",
  // cv::VideoWriter::fourcc('M','J','P','G'), 15,
  // cv::Size(output_image.size().width, output_image.size().height));
  std::cout << "before if" << std::endl;

  if (num_devices == 1) {
    int my_i = 0;
    std::chrono::time_point<std::chrono::system_clock> start_time =
        std::chrono::system_clock::now();
    while (std::chrono::duration<double>(std::chrono::system_clock::now() -
                                         start_time)
                   .count() < greenscreen_duration &&
           !exit_flag) {
      vector<k4a::capture> captures;
      // secondary_config isn't actually used here because there's no secondary
      // device but the function needs it
      captures = capturer.get_synchronized_captures(secondary_config, true);
      std::cout << "new capture" << std::endl;
      k4a::image main_color_image = captures[0].get_color_image();
      k4a::image main_depth_image = captures[0].get_depth_image();

      // let's green screen out things that are far away.
      // first: let's get the main depth image into the color camera space
      k4a::image main_depth_in_main_color =
          Utils::create_depth_image_like(main_color_image);
      main_depth_to_main_color.depth_image_to_color_camera(
          main_depth_image, &main_depth_in_main_color);
      cv::Mat cv_main_depth_in_main_color =
          Utils::depth_to_opencv(main_depth_in_main_color);
      cv::Mat cv_main_color_image = Utils::color_to_opencv(main_color_image);

      // single-camera case
      cv::Mat within_threshold_range =
          (cv_main_depth_in_main_color != 0) &
          (cv_main_depth_in_main_color < depth_threshold);
      // show the close details
      cv_main_color_image.copyTo(output_image, within_threshold_range);
      // hide the rest with the background image
      // background_image.copyTo(output_image, ~within_threshold_range);
      cv_main_color_image.copyTo(output_image, ~within_threshold_range);

      double min;
      double max;
      cv::minMaxIdx(cv_main_depth_in_main_color, &min, &max);
      cv::Mat adjMap;
      // expand your range to 0..255. Similar to histEq();
      cv_main_depth_in_main_color.convertTo(adjMap, CV_8UC1, 255 / (max - min),
                                            -min);

      // this is great. It converts your grayscale image into a tone-mapped one,
      // much more pleasing for the eye
      // function is found in contrib module, so include contrib.hpp
      // and link accordingly
      cv::Mat falseColorsMap;
      cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

      // my_test(main_depth_image.handle(), main_calibration, "file_single_" +
      // std::to_string(my_i) + ".ply");
      cv::imshow("Color", output_image);
      cv::imshow("Depth", falseColorsMap);
      // Utils::point_cloud_depth_to_color(main_depth_to_main_color,
      // main_depth_image, main_color_image, "file_single_" +
      // std::to_string(my_i) + ".ply"); my_i++;
      cv::waitKey(1);
      // break;
    }
  } else if (num_devices == 2) {
    // This wraps all the device-to-device details
    Transformation tr_secondary_color_to_main_color =
        Calibration::calibrate_devices(
            capturer, main_config, secondary_config, chessboard_pattern,
            chessboard_square_length, calibration_timeout);

    std::cout << "end of calibration" << std::endl;
    k4a::calibration secondary_calibration =
        capturer.get_subordinate_device_by_index(0).get_calibration(
            secondary_config.depth_mode, secondary_config.color_resolution);
    // Get the transformation from secondary depth to secondary color using its
    // calibration object
    Transformation tr_secondary_depth_to_secondary_color =
        Calibration::get_depth_to_color_transformation_from_calibration(
            secondary_calibration);

    k4a::transformation my_tr_secondary_depth_to_secondary_color(
        secondary_calibration);

    // We now have the secondary depth to secondary color transform. We also
    // have the transformation from the secondary color perspective to the main
    // color perspective from the calibration earlier. Now let's compose the
    // depth secondary -> color secondary, color secondary -> color main into
    // depth secondary -> color main
    Transformation tr_secondary_depth_to_main_color =
        tr_secondary_depth_to_secondary_color.compose_with(
            tr_secondary_color_to_main_color);

    // Construct a new calibration object to transform from the secondary depth
    // camera to the main color camera
    k4a::calibration secondary_depth_to_main_color_cal =
        Calibration::construct_device_to_device_calibration(
            main_calibration, secondary_calibration,
            tr_secondary_depth_to_main_color);

    k4a::calibration secondary_color_to_main_color_cal =
        Calibration::construct_device_to_device_color_calibration(
            main_calibration, secondary_calibration,
            tr_secondary_color_to_main_color);

    k4a::transformation secondary_depth_to_main_color(
        secondary_depth_to_main_color_cal);
    std::cout << "before while" << std::endl;
    std::vector<Utils::my_capture> capture_list;
    int my_i = 0;
    // sleep(3);
    for ( int k = 0 ; k < 10 ; k++){
        std::cout << k << std::endl;
        for ( int k2 = 0 ; k2 < 10000 ; k2++){
            for ( int k3 = 0 ; k3 < 10000 ; k3++);
        }
    }
    std::chrono::time_point<std::chrono::system_clock> start_time =
        std::chrono::system_clock::now();
    while (std::chrono::duration<double>(std::chrono::system_clock::now() -
                                         start_time)
                   .count() < greenscreen_duration &&
           !exit_flag && my_i < 45) {
      vector<k4a::capture> captures;
      captures = capturer.get_synchronized_captures(secondary_config, true);
      std::cout << "new capture" << std::endl;
      k4a::image main_color_image = captures[0].get_color_image();
      k4a::image main_depth_image = captures[0].get_depth_image();

      // // let's green screen out things that are far away.
      // // first: let's get the main depth image into the color camera space
      // k4a::image main_depth_in_main_color =
      //     Utils::create_depth_image_like(main_color_image);
      // main_depth_to_main_color.depth_image_to_color_camera(
      //     main_depth_image, &main_depth_in_main_color);
      // cv::Mat cv_main_depth_in_main_color =
      //     Utils::depth_to_opencv(main_depth_in_main_color);
      // cv::Mat cv_main_color_image = Utils::color_to_opencv(main_color_image);

      k4a::image secondary_color_image = captures[1].get_color_image();
      k4a::image secondary_depth_image = captures[1].get_depth_image();

      Utils::my_capture current_capture;
      current_capture.main = k4a::capture(captures[0]);
      current_capture.secondary = k4a::capture(captures[1]);
      capture_list.push_back(current_capture);
      my_i++;

      // std::cout << "before secondary color to secondary depth tr" <<
      // std::endl; k4a::image transformed_secondary_color_image =
      // k4a::image::create(
      //     K4A_IMAGE_FORMAT_COLOR_BGRA32,
      //     secondary_depth_image.get_width_pixels(),
      //     secondary_depth_image.get_height_pixels(),
      //     secondary_depth_image.get_width_pixels() * (int)sizeof(int32_t));
      // tr_secondary_color_to_secondary_depth.color_image_to_depth_camera(
      //     secondary_depth_image, secondary_color_image,
      //     &transformed_secondary_color_image);

      // // Get the depth image in the main color perspective
      // k4a::image secondary_depth_in_main_color =
      //     Utils::create_depth_image_like(main_color_image);
      // secondary_depth_to_main_color.depth_image_to_color_camera(
      //     secondary_depth_image, &secondary_depth_in_main_color);
      // cv::Mat cv_secondary_depth_in_main_color =
      //     Utils::depth_to_opencv(secondary_depth_in_main_color);

      // // Now it's time to actually construct the green screen. Where the
      // depth
      // // is 0, the camera doesn't know how far away the object is because it
      // // didn't get a response at that point. That's where we'll try to fill
      // in
      // // the gaps with the other camera.
      // cv::Mat main_valid_mask = cv_main_depth_in_main_color != 0;
      // cv::Mat secondary_valid_mask = cv_secondary_depth_in_main_color != 0;
      // // build depth mask. If the main camera depth for a pixel is valid and
      // the
      // // depth is within the threshold, then set the mask to display that
      // pixel.
      // // If the main camera depth for a pixel is invalid but the secondary
      // depth
      // // for a pixel is valid and within the threshold, then set the mask to
      // // display that pixel.
      // cv::Mat within_threshold_range =
      //     (main_valid_mask & (cv_main_depth_in_main_color < depth_threshold))
      //     |
      //     (~main_valid_mask & secondary_valid_mask &
      //      (cv_secondary_depth_in_main_color < depth_threshold));
      // // copy main color image to output image only where the mask
      // // within_threshold_range is true
      // cv_main_color_image.copyTo(output_image, within_threshold_range);
      // // fill the rest with the background image
      // // background_image.copyTo(output_image, ~within_threshold_range);
      // black_background.copyTo(output_image, ~within_threshold_range);

      // Utils::my_test(main_depth_image.handle(), main_calibration,
      // "pc_camera1_raw_" + std::to_string(my_i) + ".ply");
      // Utils::my_test(secondary_depth_image.handle(), secondary_calibration,
      // "pc_camera2_raw_" + std::to_string(my_i) + ".ply");
      // Utils::my_test(main_depth_in_main_color.handle(), main_calibration,
      // "pc_camera1_transformed_" + std::to_string(my_i) + ".ply");
      // Utils::my_test(secondary_depth_in_main_color.handle(),
      // secondary_calibration, "pc_camera2_transformed_" + std::to_string(my_i)
      // + ".ply"); my_i++; video.write(output_image);

      // if (enable_visualization) {
      //   double min;
      //   double max;
      //   cv::minMaxIdx(cv_secondary_depth_in_main_color, &min, &max);
      //   cv::Mat adjMap;
      //   // expand your range to 0..255. Similar to histEq();
      //   cv_secondary_depth_in_main_color.convertTo(adjMap, CV_8UC1,
      //                                              255 / (max - min), -min);

      //   // this is great. It converts your grayscale image into a tone-mapped
      //   // one, much more pleasing for the eye function is found in contrib
      //   // module, so include contrib.hpp and link accordingly
      //   cv::Mat falseColorsMap;
      //   cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);

      // cv::imshow("Color", cv_main_color_image);
      //   cv::imshow("Depth", falseColorsMap);
      // }

      // Utils::point_cloud_depth_to_color(
      //     main_depth_to_main_color, main_depth_image, main_color_image,
      //     "main_camera/pc_transformed_" + std::to_string(my_i) + ".ply");

      // Utils::point_cloud_depth_to_color(
      //     my_tr_secondary_depth_to_secondary_color, secondary_depth_image,
      //     secondary_color_image,
      //     "secondary_camera/pc_transformed_" + std::to_string(my_i) + ".ply",
      //     &tr_secondary_color_to_main_color);
      // Utils::point_cloud_custom(
      //     secondary_depth_to_main_color, secondary_depth_image,
      //     secondary_color_image, main_color_image,
      //     "secondary_camera/pc_transformed_" + std::to_string(my_i) +
      //     ".ply");
      // cv::waitKey(1);
      // break;
    }

    capturer.close_devices();

    for (int i = 0; i < capture_list.size(); i++) {
      Utils::my_capture current_capture = capture_list[i];
      k4a::image main_color_image = current_capture.main.get_color_image();
      k4a::image main_depth_image = current_capture.main.get_depth_image();
      k4a::image secondary_color_image =
          current_capture.secondary.get_color_image();
      k4a::image secondary_depth_image =
          current_capture.secondary.get_depth_image();

      Utils::point_cloud_depth_to_color(
          main_depth_to_main_color, main_depth_image, main_color_image,
          "main_camera/pc_transformed_" + std::to_string(i) + ".ply");

      Utils::point_cloud_depth_to_color(
          my_tr_secondary_depth_to_secondary_color, secondary_depth_image,
          secondary_color_image,
          "secondary_camera/pc_transformed_" + std::to_string(i) + ".ply",
          &tr_secondary_color_to_main_color);
    }

  } else {
    cerr << "Invalid number of devices!" << endl;
    exit(1);
  }
  // video.release();

  return 0;
}
