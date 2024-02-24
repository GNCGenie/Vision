#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;
using namespace aruco;

int main(int argv, char** argc)
{
    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    namedWindow("ArUco Marker Detection", WINDOW_NORMAL);
    resizeWindow("ArUco Marker Detection", 640, 480);

    // Create ArUco dictionary
    Dictionary dict = getPredefinedDictionary(DICT_4X4_50);
    Ptr<Dictionary> dictPtr = Ptr<Dictionary>(&dict);

    while (true) {
        // Read a frame from the video stream
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

               // Detect ArUco markers in the frame
               vector<int> markerIds;
               vector<vector<Point2f>> markerCorners;
               detectMarkers(frame, dictPtr, markerCorners, markerIds);

        // Display the frame
        imshow("ArUco Marker Detection", frame);

        // Exit the loop when the 'ESC' key is pressed
        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    return 0;
}
