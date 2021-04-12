#include <iostream>
#include <string>
#include <chrono>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <l2norm_helper.h>
#include <opencv2/highgui.hpp>
#include "faceNet.h"
#include "videoStreamer.h"
#include "network.h"
#include "Haar.h"
#include <sstream>
#include <vector>
// Uncomment to print timings in milliseconds
// #define LOG_TIMES
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace nvinfer1;
using namespace nvuffparser;
using namespace cv::ml;
vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}
int main()
{
    Logger gLogger = Logger();
    // Register default TRT plugins (e.g. LRelu_TRT)
    if (!initLibNvInferPlugins(&gLogger, "")) { return 1; }

    // USER DEFINED VALUES
    const string uffFile="../facenetModels/facenet.uff";
    const string engineFile="../facenetModels/facenet.engine";
    // DataType dtype = DataType::kHALF;
    // DataType dtype = DataType::kFLOAT;
    bool serializeEngine = true;
    int batchSize = 1;
    int nbFrames = 0;
    int videoFrameWidth = 640;
    int videoFrameHeight = 480;
    int maxFacesPerScene = 5;
    float knownPersonThreshold = 1.;
    bool isCSICam = false;

    // init facenet
    FaceNetClassifier faceNet = FaceNetClassifier(gLogger, uffFile, engineFile, batchSize, serializeEngine,
            knownPersonThreshold, maxFacesPerScene, videoFrameWidth, videoFrameHeight);

    // init opencv stuff
    VideoStreamer videoStreamer = VideoStreamer(0, videoFrameWidth, videoFrameHeight, 60, isCSICam);
    cv::Mat frame;

    // init Haar
    Haar Haar(videoFrameHeight, videoFrameWidth);

    //init Bbox and allocate memory for "maxFacesPerScene" faces per scene
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);
    int lb = 0;
    // get embeddings of known faces
    std::vector<struct Paths> paths;
    cv::Mat image;
    std::string label;
    getFilePaths("../imgs", paths);
    cout<<paths.size()<<endl;
    float embedding[paths.size()][128]; 
    int labels[paths.size()]; 
    for(int i=0; i < paths.size(); i++) {
        cout<<paths[i].absPath<<endl;
        loadInputImage(paths[i].absPath, image, 160, 160);
        vector<string> v = split (paths[i].absPath, "/");
        v.pop_back();
        label=v.back();
        std::cout<<label<<endl;
        faceNet.getE(image,embedding[i]);
        stringstream geek(label);      
        geek >> lb;
        // cout<<"ngoai class"<<embedding[i][120]<<endl;
        labels[i]=lb;
    }
    outputBbox.clear();
    Mat trainingDataMat(paths.size(), 128, CV_32F, embedding);
    Mat labelsMat(paths.size(), 1, CV_32SC1, labels);
    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    svm->save("trained-svm1000.xml");
    std::cout<<"done";
    loop over frames with inference

}