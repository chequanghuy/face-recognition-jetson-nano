#ifndef MTCNN_H
#define MTCNN_H
#include "network.h"
#include "pnet_rt.h"
#include "rnet_rt.h"
#include "onet_rt.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
class mtcnn
{
public:
    mtcnn(int row, int col);
    ~mtcnn();
    vector<struct Bbox> findFace(cv::Mat &image);
private:
    CascadeClassifier face_cascade;
    
    cv::Mat reImage;
    float nms_threshold[3];
    vector<float> scales_;
    
    vector<struct Bbox> firstBbox_;
    vector<struct orderScore> firstOrderScore_;

};

#endif