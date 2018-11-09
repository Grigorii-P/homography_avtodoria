//TODO OpenCV Contrib version is needed
//TODO подчистить ненужные инклюды
#include <iostream>
#include <fstream>
#include <sys/stat.h>
// #include <math.h>
// #include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp> 

#include "MemPool.h"
#include "Common.hpp"
#include "Tracker.h"

//TODO перенести комменты из питоновского кода сюда
//TODO дописать комменты по максимому

//TODO Алмаз хотел разбить весь алгоритм на более мелкие составляющие

using namespace std;
using namespace cv;

//TODO del
const string path_img_1 = "/home/grigorii/Desktop/momentum_speed/repers/plates_cropped_myself/A283CO716@_52.jpg";
const string path_img_2 = "/home/grigorii/Desktop/momentum_speed/repers/plates_cropped_myself/A283PO71@@_61.jpg";

const int dst_size_height = 2000;
const int trees = 5; // num of randomized kdtrees to create
const int checks = 50; // num of traversals to make on each tree
const float ratio_ = 0.75;
const int FLANN_INDEX_KDTREE = 1;
const bool sorted = true; // if multiple hits during search (radius search), return the closest one
// source: https://books.google.ru/books?id=LPm3DQAAQBAJ&pg=PA576&lpg=PA576&dq=FlannBasedMatcher+parameters+c%2B%2B+example&source=bl&ots=2vLjRgfiu8&sig=O3MId6XWoEZvaRFJJbDopoLl1to&hl=ru&sa=X&ved=2ahUKEwjqn47M9bXeAhXQhKYKHUFyB3YQ6AEwCHoECAEQAQ#v=onepage&q=FlannBasedMatcher%20parameters%20c%2B%2B%20example&f=false
const int k = 2; //num of closest neighbours
const int min_match_count = 4;
const int ransacReprojThreshold = 3; // E.g. if dst_points coordinates are measured in pixels with pixel-accurate precision, 
//it makes sense to set this parameter somewhere in the range ~1..3
Ptr<Feature2D> sift_detector = xfeatures2d::SiftFeatureDetector::create();
FlannBasedMatcher matcher = FlannBasedMatcher(new flann::KDTreeIndexParams(trees), new flann::SearchParams(checks, 0, sorted));
// матрица трансформации между дорогой и экраном
float scale;
Mat global_hom;

// EventChannel* ch;

static bool fileExists(const string& file_name) {
    struct stat buffer;   
    return (stat(file_name.c_str(), &buffer) == 0); 
}

static vector<Point2f> readCoordsFromFile(const string& file_name) {
    std::fstream myfile(file_name, std::ios_base::in);
    vector<Point2f> pts;
    float first_component, second_component;
    while (myfile >> first_component)
    {
        myfile >> second_component;
        pts.push_back(cvPoint2D32f(first_component, second_component));
    }
    return std::move(pts);
}

static void calculateHomography() {
    vector<Point2f> pts_src = readCoordsFromFile("src_points");
    vector<Point2f> pts_real = readCoordsFromFile("real_points");

    float x_max = 0, x_min = 1000;
    float y_max = 0, y_min = 1000;
    for (auto pt : pts_real) {
        if (pt.x < x_min) { x_min = pt.x; }
        if (pt.x > x_max) { x_max = pt.x; }
        if (pt.y < y_min) { y_min = pt.y; }
        if (pt.y > y_max) { y_max = pt.y; }
    }

    float resolution_scale = (x_max - x_min) / (y_max - y_min);
    int dst_size_width = int(dst_size_height * resolution_scale);
    float scale_x = dst_size_width / (x_max - x_min);
    float scale_y = dst_size_height / (y_max - y_min);
    scale = (scale_x + scale_y) / 2;

    vector<Point2f> pts_dst(pts_real.size);
    for (int i = 0; i < pts_real.size; i++) {
        pts_dst[i] = cvPoint2D32f(pts_real[i].x * scale_x, pts_real[i].y * scale_y);
    }

    //TODO take another ransacReprojThreshold (different from local_home)
    //TODO change path for homography.yml
    global_hom = findHomography(pts_src, pts_dst, CV_RANSAC, ransacReprojThreshold);
    FileStorage write("homography.yml", FileStorage::WRITE);
    write << "m" << global_hom;
    write.release();
}

static vector<float> getAverageAndMedianSpeed(vector<Point2f> speed_pts, vector<uint64_t> speed_times) {
    vector<Point2f> transformed_pts(speed_pts.size());
    // находим проекции исходных точек на экране в плоскости, перпендикулярной экрану
    perspectiveTransform(speed_pts, transformed_pts, global_hom);
    Point2f src, dst;
    float dist;
    vector<float> dists;
    for (int i = 0; i < transformed_pts.size() - 1; i++) {
        src = transformed_pts[i];
        dst = transformed_pts[i + 1];
        dist = sqrt(pow(src.x - dst.x, 2) + pow(src.y - dst.y, 2));
        dists.push_back(dist / scale); // scale - коэффициент для получения метров
    }
    sort(dists.begin(), dists.end());
    stopped here

}

static void /*InstanceSpeedEvent*/ instanceSpeedResults(const TrackEvent &track_event) {
    //TODO change Frame.h for Image.h
    //TODO Алмаз создаст сообщение ("Special for InstanceSpeed exclusively", Track track)

    if (!track_event.tracks) {
        return; //TODO return -1
    }

    auto& track = track_event.tracks->at(track_event.track_id);
    const wstring& number = track.number;

    vector<KeyPoint> keypoints_src;
    // массив реперной точки на всех плашках конкретного номера
    vector<Point2f> speed_pts;
    // массив времени соответствующих фреймов (наносекунды)
    vector<uint64_t> speed_times;
    Mat descriptors_src;

    // находим ключевые точки для првого фрейма (src), 
    // далее в цикле находим keypoints_dst, descriptors_dst для последующих фреймов 
    // а keypoints_src, descriptors_src только обновляем
    TrackPoint& src = track.points[0];
    // берем весь фрейм
    Mat src_frame = Mat(src.frame->img.height, src.frame->img.width, CV_8U, src.frame->img.data.get());
    // выделяем только плашку из всего фрейма
    Mat src_plate = Mat(src_frame, src.area);
    // находим ключевые точки
    sift_detector->detectAndCompute(src_plate, noArray(), keypoints_src, descriptors_src);
    // реперная точка и точка, которая будет соответствовать ей на следующем кадре
    vector<Point2f> point_to_be_transformed(1), pt_dst(1), abs_coord(1);
    //TODO check width and height correctness
    point_to_be_transformed[0] = cvPoint2D32f(src.area.width / 2, src.area.height / 2);
    abs_coord[0] = cvPoint2D32f(src.area.x + src.area.width / 2, src.area.y + src.area.height / 2);
    
    speed_pts.push_back(abs_coord[0]);
    speed_times.push_back(src.frame->timestamp);

    // для всех последовательных пар в треке сохраняем расстояния и скорость
    for (uint32_t i = 1; i < track.size; i++) {
        TrackPoint dst = track.points[i];
        Mat dst_frame = Mat(dst.area.width, dst.area.height, CV_8U, dst.frame->img.data.get());
        Mat dst_plate = Mat(dst_frame, dst.area);

        //TODO можно ли это объявление вынести из for loop?
        vector<KeyPoint> keypoints_dst;
        Mat descriptors_dst;
        vector<vector<DMatch>> knn_matches;
        vector<DMatch> good_matches;

        sift_detector->detectAndCompute(dst_plate, noArray(), keypoints_dst, descriptors_dst);

        // берем похожие точки
        matcher.knnMatch(descriptors_src, descriptors_dst, knn_matches, k);

        // выделяем наилучшие пары схожих точек
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_ * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        size_t size_good_matches = good_matches.size();
        if (size_good_matches > min_match_count) {
            vector<Point2f> src_good_matches;
            vector<Point2f> dst_good_matches;
            
            // точки, кот. понадобятся для построение матрицы трансформации между двумя плоскостями
            for(int i = 0; i < size_good_matches; i++) {
                src_good_matches.push_back(keypoints_src[good_matches[i].queryIdx].pt);
                dst_good_matches.push_back(keypoints_dst[good_matches[i].trainIdx].pt);
            }

            // матрица трансформации между номерными плашками
            Mat local_hom = findHomography(src_good_matches, dst_good_matches, CV_RANSAC, ransacReprojThreshold);
            // находим проекцию нужной точки в плоскости другой картинки
            perspectiveTransform(point_to_be_transformed, pt_dst, local_hom);
        }
        keypoints_src = std::move(keypoints_dst);
        descriptors_src = std::move(descriptors_dst);
        abs_coord[0] = cvPoint2D32f(dst.area.x + pt_dst[0].x, dst.area.y + pt_dst[0].y);
        point_to_be_transformed = std::move(pt_dst);
        
        speed_pts.push_back(abs_coord[0]);
        speed_times.push_back(dst.frame->timestamp);

    }

    speeds = getAverageAndMedianSpeed();

    // //TODO вынеси в отдельный файл InstanceSpeed.hpp
    // struct InstanceSpeedEvent {
    //     float speed_median;
    //     float speed_average;
    //     wstring number;
    // } speed(speed_value, number);
    // ch->publish_data("InstanceSpeed", speed);

}


// extern "C" {

// //    using namespace tracker;

//     uint32_t init(InitData& data) {
//         ch = &data.channel;
//         auto& conf_ = data.conf;

//         if (conf_.is_null()) {
//             ch->publish_data("Error", Message{"InstanceSpeed", "Conf is missing"});
//             return 1;
//         }

//         if (auto err = setup_conf(conf_); err)
//             return err;

//         {
//             Subscriber sb;
//             sb.name = "InstanceSpeed";
//             sb.action = [] (SmartPtr<void> data) {
//                 active.send([plates = data.as<Plates>()] {
// //                    auto
//                     processTrack(plates);
//                 });
//             };

//             ch->subscribe("New recognized plates", sb);
//         }

//         return Success;


//     }

// }


// g++ -O3 -std=c++11 instance_speed.cpp -o instance_speed $(pkg-config --libs opencv) && ./instance_speed
int main(int argc, const char* argv[]) {
    // Mat m(3, 3, CV_8UC1);
    // randu(m, 0, 1000);

    // FileStorage read("fs.yml", FileStorage::READ);
    // Mat m1;
    // read["m"] >> m1;
    // read.release();

    // cout << m1 << endl;

    





    // Mat im_1 = cv::imread(path_img_1, 0);
    // Mat im_2 = cv::imread(path_img_2, 0);
    // Mat output;
    
    // vector<cv::KeyPoint> keypoints_src, keypoints_dst;
    // Mat descriptors_src, descriptors_dst;
    // vector<vector<DMatch>> knn_matches;
    // vector<DMatch> good_matches;

    // sift_detector->detectAndCompute(im_1, noArray(), keypoints_src, descriptors_src);
    // sift_detector->detectAndCompute(im_2, noArray(), keypoints_dst, descriptors_dst);

    // matcher.knnMatch(descriptors_src, descriptors_dst, knn_matches, 2);

    // // https://docs.opencv.org/3.4.3/d5/d6f/tutorial_feature_flann_matcher.html
    // for (size_t i = 0; i < knn_matches.size(); i++) {
    //     if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
    //         good_matches.push_back(knn_matches[i][0]);
    //     }
    // }
    // // https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html
    // // for( int i = 0; i < descriptors_src.rows; i++ ) { 
    // //     if(matches[i].distance <= max(2 * min_dist, 0.02)) { 
    // //         good_matches.push_back( matches[i]); 
    // //     }
    // // }
    
    // size_t size_good_matches = good_matches.size();
    // if (size_good_matches > min_match_count) {
    //     // drawMatches(im_1, keypoints_src, im_2, keypoints_dst, good_matches, output, Scalar::all(-1),
    //     //          Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //     vector<Point2f> src;
    //     vector<Point2f> dst;
    //     for(int i = 0; i < size_good_matches; i++) {
    //         src.push_back(keypoints_src[good_matches[i].queryIdx].pt);
    //         dst.push_back(keypoints_dst[good_matches[i].trainIdx].pt);
    //     }
    //     Mat local_hom = findHomography(src, dst, CV_RANSAC );

    //     vector<Point2f> pt_src(1); 
    //     pt_src[0] = cvPoint(70, 15);
    //     vector<Point2f> pt_dst(1);
        
    //     perspectiveTransform(pt_src, pt_dst, local_hom);

    //     circle(im_1, pt_src[0], 2, Scalar(255,0,255), -1);
    //     circle(im_2, pt_dst[0], 2, Scalar(255,0,255), -1);
    // }

    // imshow("res.jpg", im_1);
    // waitKey(0);
    // imshow("res.jpg", im_2);
    // waitKey(0);

    return 0;
}

//TODO брать середину или же дескриптор? 
//TODO что если на первой (или промежуточной фотке) не нашли похожих дескрипторов
//TODO how can we handle more than one homography for different road lanes ???
//TODO may also try MSER OpenCV algorithm instead of cv.FlannBasedMatcher (dropbox uses it in text detection)