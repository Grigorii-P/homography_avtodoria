//TODO OpenCV Contrib version is needed
#include "Tracker.h"
#include "MemPool.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp> 
#include <iostream>

using namespace std;
using namespace cv;

const string path_img_1 = "/home/grigorii/Desktop/momentum_speed/repers/plates_cropped_myself/A283CO716@_52.jpg";
const string path_img_2 = "/home/grigorii/Desktop/momentum_speed/repers/plates_cropped_myself/A283PO71@@_61.jpg";

const int trees = 5; // num of randomized kdtrees to create
const int checks = 50; // num of traversals to make on each tree
const float ratio = 0.75;
const int FLANN_INDEX_KDTREE = 1;
const bool sorted = true; // if multiple hits during search (radius search), return the closest one
// source: https://books.google.ru/books?id=LPm3DQAAQBAJ&pg=PA576&lpg=PA576&dq=FlannBasedMatcher+parameters+c%2B%2B+example&source=bl&ots=2vLjRgfiu8&sig=O3MId6XWoEZvaRFJJbDopoLl1to&hl=ru&sa=X&ved=2ahUKEwjqn47M9bXeAhXQhKYKHUFyB3YQ6AEwCHoECAEQAQ#v=onepage&q=FlannBasedMatcher%20parameters%20c%2B%2B%20example&f=false
const int k = 2; //num of closest neighbours
const int min_match_count = 4;
const int ransacReprojThreshold = 3; // E.g. if dst_points coordinates are measured in pixels with pixel-accurate precision, 
//it makes sense to set this parameter somewhere in the range ~1..3
Ptr<Feature2D> sift_detector = xfeatures2d::SiftFeatureDetector::create();
FlannBasedMatcher matcher = FlannBasedMatcher(new flann::KDTreeIndexParams(trees), new flann::SearchParams(checks, 0, sorted));

static void calculate_homography() {}

static void instance_speed(const TrackEvent &track_event) {
    //TODO std_move
    //TODO change Frame.h for Image.h

    // message I need: ("Track Finalized", TrackEvent{nullptr, track_id})
    // but I also need to get &tracks from ("New Track", TrackEvent{&tracks, track_id}) or ("Track Updated", TrackEvent{&tracks, track_id})
    wstring number = track_event.tracks->at(track_event.track_id).number;

    //TODO dont forget
    // находим ключевые точки для првого фрейма, 
    // далее в цикле находим keypoints_dst, descriptors_dst для последующих фреймов 
    // а keypoints_src, descriptors_src только обновляем
    vector<KeyPoint> keypoints_src;
    Mat descriptors_src;

    TrackPoint src = track_event.tracks->at(track_event.track_id).points[0];
    // берем весь фрейм
    Mat src_frame = Mat(src.area.width, src.area.height, CV_8U, (void*)src.frame->data.operator*);
    // выделяем только плашку из всего фрейма
    Mat src_plate = Mat(src_frame, src.area);
    // находим ключевые точки
    sift_detector->detectAndCompute(src_plate, noArray(), keypoints_src, descriptors_src);

    // для всех последовательных пар в треке считаем расстояния и скорость
    for (uint32_t i = 1; i < track_event.tracks->at(track_event.track_id).size - 1; i++) {
        TrackPoint dst = track_event.tracks->at(track_event.track_id).points[i];
        //TODO check relative correctness of width and height
        Mat dst_frame = Mat(dst.area.width, dst.area.height, CV_8U, (void*)dst.frame->data.operator*);
        Mat dst_plate = Mat(dst_frame, dst.area);

        //TODO можно ли это объявление вынести из for loop?
        vector<KeyPoint> keypoints_dst;
        Mat descriptors_dst;
        vector<vector<DMatch>> knn_matches;
        vector<DMatch> good_matches;

        sift_detector->detectAndCompute(dst_plate, noArray(), keypoints_dst, descriptors_dst);

        // берем похожие точки
        matcher.knnMatch(descriptors_src, descriptors_dst, knn_matches, 2);

        // выделяем наилучшие пары схожих точек
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
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
            
            // матрица трансформации
            Mat local_hom = findHomography(src_good_matches, dst_good_matches, CV_RANSAC );

            vector<Point2f> pt_src(1), pt_dst(1);
            pt_src[0] = cvPoint(, );
            // находим проекцию нужной точки в плоскости другой картинки
            perspectiveTransform(pt_src, pt_dst, local_hom);

        }

        //TODO обновить keypoints_src, descriptors_src
    }
}

int main(int argc, const char* argv[]) {
    
    Mat im_1 = cv::imread(path_img_1, 0);
    Mat im_2 = cv::imread(path_img_2, 0);
    Mat output;
    
    vector<cv::KeyPoint> keypoints_src, keypoints_dst;
    Mat descriptors_src, descriptors_dst;
    vector<vector<DMatch>> knn_matches;
    vector<DMatch> good_matches;

    sift_detector->detectAndCompute(im_1, noArray(), keypoints_src, descriptors_src);
    sift_detector->detectAndCompute(im_2, noArray(), keypoints_dst, descriptors_dst);

    matcher.knnMatch(descriptors_src, descriptors_dst, knn_matches, 2);

    // https://docs.opencv.org/3.4.3/d5/d6f/tutorial_feature_flann_matcher.html
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    // https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html
    // for( int i = 0; i < descriptors_src.rows; i++ ) { 
    //     if(matches[i].distance <= max(2 * min_dist, 0.02)) { 
    //         good_matches.push_back( matches[i]); 
    //     }
    // }
    
    size_t size_good_matches = good_matches.size();
    if (size_good_matches > min_match_count) {
        // drawMatches(im_1, keypoints_src, im_2, keypoints_dst, good_matches, output, Scalar::all(-1),
        //          Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        vector<Point2f> src;
        vector<Point2f> dst;
        for(int i = 0; i < size_good_matches; i++) {
            src.push_back(keypoints_src[good_matches[i].queryIdx].pt);
            dst.push_back(keypoints_dst[good_matches[i].trainIdx].pt);
        }
        Mat local_hom = findHomography(src, dst, CV_RANSAC );

        vector<Point2f> pt_src(1); 
        pt_src[0] = cvPoint(70, 15);
        vector<Point2f> pt_dst(1);
        
        perspectiveTransform(pt_src, pt_dst, local_hom);

        circle(im_1, pt_src[0], 2, Scalar(255,0,255), -1);
        circle(im_2, pt_dst[0], 2, Scalar(255,0,255), -1);
    }

    imshow("res.jpg", im_1);
    waitKey(0);
    imshow("res.jpg", im_2);
    waitKey(0);

    return 0;
    // g++ -O3 -std=c++11 main.cpp -o main $(pkg-config --libs opencv) & ./main
}

//TODO брать середину или же дескриптор? 
//TODO что если на первой (или промежуточной фотке) не нашли похожих дескрипторов
//TODO how can we handle more than one homography for different road lanes ???
//TODO may also try MSER OpenCV algorithm instead of cv.FlannBasedMatcher (dropbox uses it in text detection)