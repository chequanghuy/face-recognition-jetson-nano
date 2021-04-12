#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
  
#include <iostream>
#include <stdio.h>
  
using namespace std;
using namespace cv;
  
int main( )
{
    Mat image;
    image = imread("/home/huycq/AlphaPose/examples/demo/2.jpg", CV_LOAD_IMAGE_COLOR);  
    namedWindow( "window1", 1 );   
    imshow( "window1", image );
  
    // Load Face cascade (.xml file)
    CascadeClassifier face_cascade;
    face_cascade.load( "/home/huycq/Downloads/haarcascade_frontalface_default.xml" );

 if(face_cascade.empty())
// if(!face_cascade.load("D:\\opencv2410\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"))
 {
  cerr<<"Error Loading XML file"<<endl;
  return 0;
 }
 
    // Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE);
  
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {
        // Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        // ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        rectangle( image, Point( faces[i].x, faces[i].y ), Point( faces[i].x+ faces[i].width, faces[i].y+ faces[i].height), Scalar( 0, 255, 255 ), LINE_8 );
    }
      
    imshow( "Detected Face", image );
      
    waitKey(0);                   
    return 0;
}