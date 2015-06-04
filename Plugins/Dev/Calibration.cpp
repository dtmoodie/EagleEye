#include "Calibration.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

RUNTIME_COMPILER_LINKLIBRARY("-lopencv_calib3d")

using namespace EagleLib;

void CalibrateCamera::Init(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Clear", boost::bind(&CalibrateCamera::clear, this));
    updateParameter("Num corners X", 6);
    updateParameter("Num corners Y", 9);
    updateParameter("Corner distance (mm)", double(18.75));
    updateParameter("Min pixel distance", 10.0);
    lastCalibration = 0;

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
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
    }
    TIME
    img.download(h_img, stream);
    TIME
    stream.waitForCompletion();
    TIME
    cv::Mat h_mat = h_img.createMatHeader();
    cv::Mat corners;
    TIME
    bool found = cv::findChessboardCorners(h_mat, cv::Size(numX, numY), corners);
    cv::drawChessboardCorners(h_mat,cv::Size(numX,numY),corners, found);
    UIThreadCallback::getInstance().addCallback(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow),fullTreeName, h_mat));
    //UIThreadCallback::getInstance().addCallback(boost::bind(&cv::imshow, fullTreeName, h_mat));
    //cv::imshow(fullTreeName, h_mat);
    TIME
    boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);
    if(corners.rows == objectPoints3d.size() && found)
    {
        cv::Vec2f centroid(0,0);
        for(int i = 0; i < corners.rows; ++i)
        {
            centroid += corners.at<cv::Vec2f>(i);
        }
        centroid /= float(corners.rows);

        float minDist = std::numeric_limits<float>::max();
        for(cv::Vec2f& other: imagePointCentroids)
        {
            float dist = cv::norm(other - centroid);
            if(dist < minDist)
                minDist = dist;
        }
        if(minDist > getParameter<double>("Min pixel distance")->data)
        {
            imagePointCollection.push_back(corners);
            objectPointCollection.push_back(objectPoints3d);
            imagePointCentroids.push_back(centroid);
            cv::Mat K;
            cv::Mat distortionCoeffs;
            std::vector<cv::Mat> rvecs;
            std::vector<cv::Mat> tvecs;
            if(objectPointCollection.size() > lastCalibration + 10)
            {
                cv::calibrateCamera(objectPointCollection, imagePointCollection,
                                    img.size(), K, distortionCoeffs,rvecs,tvecs);
                updateParameter("Camera matrix", K);
                updateParameter("Distortion matrix", distortionCoeffs);
                lastCalibration = objectPointCollection.size();
            }
        }
    }

    return img;
}

void CalibrateStereoPair::Init(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Clear", boost::bind(&CalibrateCamera::clear, this));
    updateParameter("Num corners X", 6);
    updateParameter("Num corners Y", 9);
    updateParameter("Corner distance (mm)", double(18.75));
    updateParameter("Min pixel distance", 10.0);
    lastCalibration = 0;
}

cv::cuda::GpuMat CalibrateStereoPair::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateCamera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateStereoPair)
