cmake_minimum_required (VERSION 3.9.0)
project (CAMERA_SERVICE)

set (K4A_INCLUDE_PATH "C:/Program Files/Azure Kinect SDK v1.2.0/sdk/include")
set (K4A_LINK_PATH "C:/Program Files/Azure Kinect SDK v1.2.0/sdk/windows-desktop/amd64/release/lib")

find_package (OpenCV CONFIG PATHS "C:/opencv/build/x64/vc14/lib" ${OpenCV_REQ_FLAG})
if (OpenCV_FOUND)
    file(COPY ${OpenCV_DIR}/../bin/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES_MATCHING PATTERN "*.dll")
    message(STATUS "opencv found")
endif ()

set (SRC_FILES utils.cpp calibration.cpp main.cpp)
set (HEADER_FILES MultiDeviceCapturer.h transformation.h calibration.h utils.h)

add_executable (camera_service ${SRC_FILES})
target_include_directories (camera_service PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${K4A_INCLUDE_PATH})
target_link_libraries (camera_service LINK_PUBLIC ${K4A_LINK_PATH}/k4a.lib ${OpenCV_LIBS} )