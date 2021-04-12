
#include "Haar.h"
// #define LOG
Haar::Haar(int row, int col){
    //set NMS thresholds
    face_cascade.load( "/home/huycq/Downloads/haarcascade_frontalface_default.xml" );
    if(face_cascade.empty())
// if(!face_cascade.load("D:\\opencv2410\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"))
    {
        cerr<<"Error Loading XML file"<<endl;
    }
}

Haar::~Haar(){
    //delete []simpleFace_;
}

vector<struct Bbox> Haar::findFace(cv::Mat &image){
    firstBbox_.clear();
    firstOrderScore_.clear();
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE);
    Bbox emty;
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {
        // Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        // ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        // rectangle( image, Point( faces[i].x, faces[i].y ), Point( faces[i].x+ faces[i].width, faces[i].y+ faces[i].height), Scalar( 0, 255, 255 ), LINE_8 );
        emty.x1=faces[i].y;
        emty.x2=faces[i].height+faces[i].y;
        emty.y1=faces[i].x;
        emty.y2=faces[i].width+faces[i].x;
        firstBbox_.push_back(emty);
    }
    return firstBbox_;
    
}

