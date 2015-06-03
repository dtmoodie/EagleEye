#include "Calibration.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

RUNTIME_COMPILER_LINKLIBRARY("-lopencv_calib3d")

using namespace EagleLib;

void CalibrateCamera::Init(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Clear", boost::bind(&CalibrateCamera::clear, this));
    updateParameter("Num corners X", 5);
    updateParameter("Num corners Y", 8);
    updateParameter("Corner distance (mm)", double(25));

}
void CalibrateCamera::clear()
{
    boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);
    objectPointCollection.clear();
    imagePointCollection.clear();
}

cv::cuda::GpuMat CalibrateCamera::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    int numX = getParameter<int>(1)->data;
    int numY = getParameter<int>(2)->data;
    if(parameters[1]->changed || parameters[2]->changed || parameters[3]->changed || objectPoints3d.size() == 0)
    {
        double dx = getParameter<double>(3)->data;
        clear();
        objectPoints3d.clear();
        for(int i = 0; i < numX; ++i)
        {
            for(int j = 0; j < numY; ++j)
            {
                objectPoints3d.push_back(cv::Point3f(dx*i, dx*j, 0));
            }
        }
    }
    TIME
    img.download(h_img, stream);
    TIME
    stream.waitForCompletion();
    TIME
    cv::Mat h_mat = h_img.createMatHeader();
    std::vector<cv::Point2f> corners;
    TIME
    bool found = cv::findChessboardCorners(h_mat, cv::Size(numX, numY), corners);
    cv::drawChessboardCorners(h_mat,cv::Size(numX,numY),corners, found);
    cv::imshow(fullTreeName, h_mat);
    TIME
    boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);
    if(corners.size() == objectPoints3d.size() && found)
    {
        imagePointCollection.push_back(corners);
        objectPointCollection.push_back(objectPoints3d);

        cv::Mat K;
        cv::Mat distortionCoeffs;
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;

        cv::calibrateCamera(objectPointCollection, imagePointCollection,
                            img.size(), K, distortionCoeffs,rvecs,tvecs);
        updateParameter("Camera matrix", K);
        updateParameter("Distortion matrix", distortionCoeffs);
    }

    return img;
}

void CalibrateStereoPair::Init(bool firstInit)
{

}

cv::cuda::GpuMat CalibrateStereoPair::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateCamera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateStereoPair)
