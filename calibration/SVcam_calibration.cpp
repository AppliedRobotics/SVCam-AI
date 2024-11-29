#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <tuple>
#include <string>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

//load images from existing
std::tuple<std::vector<cv::Mat>, std::vector<cv::Mat>, std::vector<std::string>, std::vector<std::string>> 
load_images_from_folder(const std::string& folder_path_left, const std::string& folder_path_right) {

    std::vector<cv::Mat> images_left, images_right;
    std::vector<std::string> images_name_left, images_name_right;

   

    auto it_left = fs::directory_iterator(folder_path_left);
    auto it_right = fs::directory_iterator(folder_path_right);


    while (it_left != fs::end(it_left) && it_right != fs::end(it_right)) {
        auto file_path_left = it_left->path().string();

        auto file_path_right = it_right->path().string();

        if (fs::is_regular_file(file_path_left) && fs::is_regular_file(file_path_right)) {

            cv::Mat image_left = cv::imread(file_path_left);
            cv::Mat image_right = cv::imread(file_path_right);


            if (image_left.empty()) {
                std::cerr << "Error: Could not load left image: " << file_path_left << std::endl;
                ++it_left;
                ++it_right;
                continue;
            }

            if (image_right.empty()) {
                std::cerr << "Error: Could not load right image: " << file_path_right << std::endl;
                ++it_left;
                ++it_right;
                continue;
            }

                images_name_left.push_back(file_path_left);
                images_name_right.push_back(file_path_right);

                images_left.push_back(image_left);
                images_right.push_back(image_right);

        }
        ++it_right;
        ++it_left;
    }
    return {images_left, images_right, images_name_left, images_name_right};
}

void save_images_from_cameras(const std::string& folder_left, const std::string& folder_right,
                              cv::VideoCapture& capture_left, cv::VideoCapture& capture_right, int image_count) {
    int count_image = 0;

    while (count_image < image_count) {
        cv::Mat img_left, img_right;
        bool rc_l = capture_left.read(img_left);
        bool rc_r = capture_right.read(img_right);

        if (!rc_l || img_left.empty()) {
            std::cerr << "Error: Failed to capture image from left camera." << std::endl;
            continue;
        }

        if (!rc_r || img_right.empty()) {
            std::cerr << "Error: Failed to capture image from right camera." << std::endl;
            continue;
        }

        std::string filename_left = folder_left + "/" + std::to_string(count_image) + ".png";
        std::string filename_right = folder_right + "/" + std::to_string(count_image) + ".png";

        cv::imwrite(filename_left, img_left);
        std::cout << "Saved "  << " image " << count_image << " to " << filename_left << std::endl;
        cv::imwrite(filename_right, img_right);
        std::cout << "Saved " <<  " image " << count_image << " to " << filename_right << std::endl;

        count_image++;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

void clear_directory(const std::string& path) {
    for (const auto& entry : fs::directory_iterator(path)) {
        fs::remove_all(entry.path());
    }
}

void save_corners(const std::vector<std::vector<cv::Point2f>>& corners, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "corners" << corners;
    fs.release();
    std::cout << "Saved corners to " << filename << std::endl;
}

void save_calibration(const cv::Mat& camera_matrix, const cv::Mat& dist_coefs, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "dist_coefs" << dist_coefs;
    fs.release();
    std::cout << "Saved calibration data to " << filename << std::endl;
}

void save_stereo_calibration(const cv::Mat& R, const cv::Mat& T, const cv::Mat& E, const cv::Mat& F, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    fs.release();
    std::cout << "Saved stereo calibration data to " << filename << std::endl;
}

void save_rectification(const cv::Mat& map1x, const cv::Mat& map1y, const cv::Mat& map2x, const cv::Mat& map2y, const cv::Mat& Q, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "map1x" << map1x;
    fs << "map1y" << map1y;
    fs << "map2x" << map2x;
    fs << "map2y" << map2y;
    fs << "Q" << Q;
    fs.release();
    std::cout << "Saved rectification maps to " << filename << std::endl;
}

// FIND CORNERS
std::tuple<std::vector<std::vector<cv::Point3f>>, std::vector<std::vector<cv::Point2f>>, std::vector<std::vector<cv::Point2f>>, std::vector<cv::Mat>, std::vector<cv::Mat>>
find_corners(const std::vector<cv::Mat>& images_left, const std::vector<cv::Mat>& images_right, const cv::Size& pattern_size, float square_size) {
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points_left, img_points_right;
    std::vector<cv::Mat> images_left_corners, images_right_corners;

    std::vector<cv::Point3f> pattern_points;
    for (int i = 0; i < pattern_size.height; ++i)
        for (int j = 0; j < pattern_size.width; ++j)
            pattern_points.emplace_back(j * square_size, i * square_size, 0);

    int failed_left = 0, failed_right = 0;

    for (size_t i = 0; i < images_left.size(); ++i) {
        cv::Mat gray_left, gray_right;
        cv::cvtColor(images_left[i], gray_left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(images_right[i], gray_right, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners_left, corners_right;
        bool found_left = cv::findChessboardCorners(gray_left, pattern_size, corners_left);
        bool found_right = cv::findChessboardCorners(gray_right, pattern_size, corners_right);

        if (found_left && found_right) {
            cv::cornerSubPix(gray_left, corners_left, cv::Size(11, 11), cv::Size(-1, -1), 
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            cv::cornerSubPix(gray_right, corners_right, cv::Size(11, 11), cv::Size(-1, -1), 
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

            img_points_left.push_back(corners_left);
            img_points_right.push_back(corners_right);
            obj_points.push_back(pattern_points);

            images_left_corners.push_back(images_left[i]);
            images_right_corners.push_back(images_right[i]);
        } else {
            if (!found_left) {
                std::cerr << "Corners not found in left image: " << i << std::endl;
                failed_left++;
            }
            if (!found_right) {
                std::cerr << "Corners not found in right image: " << i << std::endl;
                failed_right++;
            }
        }
    }

    // Check if enough corners were found
    if ((failed_left>=40)|| (failed_right>=40)) {
        std::cout << "Not enough corners found in more than 50% of images. Please recalibrate." << std::endl;
        return {obj_points, img_points_left, img_points_right, images_left_corners, images_right_corners};
    }
    else{
    // // Save corners to files
    save_corners(img_points_left, "left_corners.yml");
    save_corners(img_points_right, "right_corners.yml");

    return {obj_points, img_points_left, img_points_right, images_left_corners, images_right_corners};}
}

// CALIBRATE CAMERA
std::tuple<double, cv::Mat, cv::Mat, std::vector<cv::Mat>, std::vector<cv::Mat>,
           double, cv::Mat, cv::Mat, std::vector<cv::Mat>, std::vector<cv::Mat>>
calibrate_cameras(const std::vector<std::vector<cv::Point3f>>& obj_points, const std::vector<std::vector<cv::Point2f>>& img_points_left,
                  const std::vector<std::vector<cv::Point2f>>& img_points_right, const cv::Size& image_shape) {
    cv::Mat camera_matrix_left, dist_coefs_left, camera_matrix_right, dist_coefs_right;
    std::vector<cv::Mat> rvecs_left, tvecs_left, rvecs_right, tvecs_right;

    double rms_left = cv::calibrateCamera(obj_points, img_points_left, image_shape, camera_matrix_left, dist_coefs_left, rvecs_left, tvecs_left);
    double rms_right = cv::calibrateCamera(obj_points, img_points_right, image_shape, camera_matrix_right, dist_coefs_right, rvecs_right, tvecs_right);

    save_calibration(camera_matrix_left, dist_coefs_left,  "left_calibration.yml");
    save_calibration(camera_matrix_right, dist_coefs_right,  "right_calibration.yml");

    return {rms_left, camera_matrix_left, dist_coefs_left, rvecs_left, tvecs_left,
            rms_right, camera_matrix_right, dist_coefs_right, rvecs_right, tvecs_right};
}

std::tuple<double, cv::Mat, cv::Mat, cv::Mat, cv::Mat, cv::Mat, cv::Mat, cv::Mat, cv::Mat>
stereo_calibrate(const std::vector<std::vector<cv::Point3f>>& obj_points, const std::vector<std::vector<cv::Point2f>>& img_points_left,
                 const std::vector<std::vector<cv::Point2f>>& img_points_right, const cv::Mat& camera_matrix_left,
                 const cv::Mat& dist_coefs_left, const cv::Mat& camera_matrix_right, const cv::Mat& dist_coefs_right, const cv::Size& image_shape) {
    cv::Mat R, T, E, F;
    double ret = cv::stereoCalibrate(obj_points, img_points_left, img_points_right,
                                     camera_matrix_left, dist_coefs_left, camera_matrix_right, dist_coefs_right,
                                     image_shape, R, T, E, F,
                                     cv::CALIB_FIX_INTRINSIC,
                                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-5));

    save_stereo_calibration(R, T, E, F,  "stereo_calibration.yml");

    return {ret, R, T, E, F, camera_matrix_left, dist_coefs_left, camera_matrix_right, dist_coefs_right};
}

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat, cv::Mat>
stereo_rectify(const cv::Mat& camera_matrix_left, const cv::Mat& dist_coefs_left, const cv::Mat& camera_matrix_right,
               const cv::Mat& dist_coefs_right, const cv::Size& image_size, const cv::Mat& R, const cv::Mat& T) {
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validPixROI1, validPixROI2;

    cv::stereoRectify(camera_matrix_left, dist_coefs_left, camera_matrix_right, dist_coefs_right, image_size, R, T, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0, image_size, &validPixROI1, &validPixROI2);

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(camera_matrix_left, dist_coefs_left, R1, P1, image_size, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(camera_matrix_right, dist_coefs_right, R2, P2, image_size, CV_32FC1, map2x, map2y);

    save_rectification(map1x, map1y, map2x, map2y, Q, "rectification.yml");

    return {map1x, map1y, map2x, map2y, Q};
}

int main() {
    const std::string folder_left = "left_calib";
    const std::string folder_right = "right_calib";

    const std::string CAPTURE_PIPE_l = "libcamerasrc camera-name=/base/i2c@ff110000/ov4689@36 !"
                                        "video/x-raw,width=640,height=480,format=YUY2 ! videoconvert ! appsink";
    const std::string CAPTURE_PIPE_r = "libcamerasrc camera-name=/base/i2c@ff120000/ov4689@36 !"
                                        "video/x-raw,width=640,height=480,format=YUY2 ! videoconvert ! appsink";

    cv::VideoCapture capture_left(CAPTURE_PIPE_l, cv::CAP_GSTREAMER);
    cv::VideoCapture capture_right(CAPTURE_PIPE_r, cv::CAP_GSTREAMER);

    if (!capture_left.isOpened() || !capture_right.isOpened()) {
        std::cerr << "Error: Could not open camera pipeline(s)." << std::endl;
        return 1;
    }

    if (!fs::exists(folder_left) && !fs::exists(folder_right)) { 
        fs::create_directory(folder_left);
        fs::create_directory(folder_right);}

    if (fs::is_empty(folder_left) || fs::is_empty(folder_right)) {

        std::cout << "Need perform a new calibration";
            clear_directory(folder_left);
            clear_directory(folder_right);
            save_images_from_cameras(folder_left, folder_right, capture_left, capture_right, 60);
     }   
     else{
         std::string response;
        std::cout << "Do you want to perform a new calibration? (y/n): ";
        std::cin >> response;

        if (response == "y") {
            clear_directory(folder_left);
            clear_directory(folder_right);
            save_images_from_cameras(folder_left, folder_right, capture_left, capture_right, 60);

        } 

     }
    auto [images_left, images_right, images_name_left, images_name_right] = load_images_from_folder(folder_left, folder_right);

    if (images_left.empty() || images_right.empty()) {
        std::cerr << "Error: No valid images loaded for calibration. Exiting." << std::endl;
        return 1;
    }

    capture_left.release();
    capture_right.release();

    cv::Size pattern_size;
    float square_size;

    std::cout << "Enter the number of inner corners per a chessboard row (width): ";
    std::cin >> pattern_size.width;
    std::cout << "Enter the number of inner corners per a chessboard column (height): ";
    std::cin >> pattern_size.height;
    std::cout << "Enter the size of a square in your defined unit (e.g., 3 for cm): ";
    std::cin >> square_size;


    square_size = static_cast<float>(square_size) / 100.0f;

    // Find corners
    auto [obj_points, img_points_left, img_points_right, images_left_corners, images_right_corners] = 
        find_corners(images_left, images_right, pattern_size, square_size);

    // Check if enough corners were found
    if (img_points_left.size() < images_left.size() * 0.5) {
        std::cout << "Not enough corners found. Please recalibrate." << std::endl;
        return 0;
    }

    // Calibrate cameras
    auto [rms_left, camera_matrix_left, dist_coefs_left, rvecs_left, tvecs_left,
          rms_right, camera_matrix_right, dist_coefs_right, rvecs_right, tvecs_right] =
        calibrate_cameras(obj_points, img_points_left, img_points_right, images_left[0].size());

    // Stereo calibrate
    auto [ret, R, T, E, F, _, __, ___, ____] =
        stereo_calibrate(obj_points, img_points_left, img_points_right, camera_matrix_left, dist_coefs_left,
                         camera_matrix_right, dist_coefs_right, images_left[0].size());

    // Stereo rectify
    auto [map1x, map1y, map2x, map2y, Q] =
        stereo_rectify(camera_matrix_left, dist_coefs_left, camera_matrix_right, dist_coefs_right,
                       images_left[0].size(), R, T);

    return 0;
}
