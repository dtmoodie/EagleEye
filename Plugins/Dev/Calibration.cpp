#include "Calibration.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <nodes/VideoProc/Tracking.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

RUNTIME_COMPILER_LINKLIBRARY("-lopencv_calib3d")
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_highgui")
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaimgproc")
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaarithm")

using namespace EagleLib;
IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}
void CalibrateCamera::Init(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Clear", boost::bind(&CalibrateCamera::clear, this));
    updateParameter<boost::function<void(void)>>("Force calibration", boost::bind(&CalibrateCamera::calibrate, this));
    if(firstInit)
    {
        updateParameter("Num corners X", 6);
        updateParameter("Num corners Y", 9);
        updateParameter("Corner distance", double(18.75), Parameter::Control, "Distance between corners in mm");
        updateParameter("Min pixel distance", float(10.0));
        addInputParameter<TrackSparseFunctor>("Sparse tracking functor");
    }

    lastCalibration = 0;


}
void CalibrateCamera::clear()
{
    boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);
    objectPointCollection.clear();
    imagePointCollection.clear();
}
void callibrateCameraThread(std::vector<cv::Mat>& imagePointCollection, std::vector<std::vector<cv::Point3f>>& objectPointCollection)
{
    
}

cv::cuda::GpuMat CalibrateCamera::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    int numX = getParameter<int>(2)->data;
    int numY = getParameter<int>(3)->data;

    cv::Mat h_corners;
    bool found = false;
    TrackSparseFunctor* tracker = getParameter<TrackSparseFunctor*>(6)->data;

    if(parameters[2]->changed || parameters[3]->changed || parameters[4]->changed || objectPoints3d.size() == 0)
    {
        imgSize = img.size();
        double dx = getParameter<double>(4)->data;
        clear();
        objectPoints3d.clear();
        for(int i = 0; i < numY; ++i)
        {
            for(int j = 0; j < numX; ++j)
            {
                objectPoints3d.push_back(cv::Point3f(dx*j, dx*i, 0));
            }
        }
        parameters[2]->changed = false;
        parameters[3]->changed = false;
        parameters[4]->changed = false;
    }

    if(img.channels() == 3)
        cv::cuda::cvtColor(img, currentGreyFrame, cv::COLOR_BGR2GRAY, 0, stream);
    else
        currentGreyFrame = img;

    if(tracker)
    {
        if(prevFramePoints.empty())
        {
            log(Status, "Initializing with CPU corner finder routine");
            currentGreyFrame.download(h_img, stream);
            stream.waitForCompletion();

            found = cv::findChessboardCorners(h_img, cv::Size(numX, numY), corners);
            if(found)
            {
                TIME
                prevFramePoints.upload(corners, stream);
                prevFramePoints = prevFramePoints.reshape(2,1);
                prevFramePoints.copyTo(currentFramePoints, stream);
                cv::drawChessboardCorners(h_img, cv::Size(numX,numY), corners, found);
                UIThreadCallback::getInstance().addCallback(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow),fullTreeName, h_img));
                prevGreyFrame = currentGreyFrame;
                TIME
            }
        }else
        {
            TIME
            // Track points with GPU tracker
            (*tracker)(prevGreyFrame, currentGreyFrame, prevFramePoints, currentFramePoints, status, error, stream);
            // Find the centroid of the new points
            int goodPoints = cv::cuda::countNonZero(status);
            if(goodPoints == numX*numY)
            {
                log(Status, "Tracking successful with tracker");
                prevGreyFrame = currentGreyFrame;
                prevFramePoints = currentFramePoints;
                currentFramePoints.download(corners, stream);
                stream.waitForCompletion();
                h_corners = corners.createMatHeader().clone();
                found = true;
            }else
            {
                prevFramePoints.release();
            }
            TIME
        }
    }else
    {
        log(Status, "Relying on CPU corner finder routine");
        currentGreyFrame.download(h_img, stream);
        stream.waitForCompletion();

        found = cv::findChessboardCorners(h_img, cv::Size(numX, numY), corners);
        if(found)
        {
            h_corners = corners.createMatHeader();
            cv::drawChessboardCorners(h_img, cv::Size(numX,numY), corners, found);
            UIThreadCallback::getInstance().addCallback(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow),fullTreeName, h_img));
        }
        else
        {
            log(Warning, "Could not find checkerboard pattern");
        }
    }


    //UIThreadCallback::getInstance().addCallback(boost::bind(static_cast<void(*)(const cv::String&, const cv::_InputArray&)>(&cv::imshow),fullTreeName, h_mat));

    TIME
    boost::recursive_mutex::scoped_lock lock(pointCollectionMtx);

    if(h_corners.rows == objectPoints3d.size() && found)
    {
        TIME
        cv::Vec2f centroid(0,0);
        for(int i = 0; i < h_corners.rows; ++i)
        {
            centroid += h_corners.at<cv::Vec2f>(i);
        }
        centroid /= float(corners.cols);

        float minDist = std::numeric_limits<float>::max();
        for(cv::Vec2f& other: imagePointCentroids)
        {
            float dist = cv::norm(other - centroid);
            if(dist < minDist)
                minDist = dist;
        }
        TIME
        if(minDist > getParameter<float>("Min pixel distance")->data)
        {
            imagePointCollection.push_back(h_corners);
            objectPointCollection.push_back(objectPoints3d);
            imagePointCentroids.push_back(centroid);
            if(objectPointCollection.size() > lastCalibration + 10)
            {
                TIME
                calibrate();
                TIME
            }
        }
    }

    return img;
}
void CalibrateCamera::calibrate()
{
    log(Status, "Calibrating camera with " + boost::lexical_cast<std::string>(imagePointCollection.size()) + " images");
    cv::Mat K;
    cv::Mat distortionCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double quality = cv::calibrateCamera(objectPointCollection, imagePointCollection,
                        imgSize, K, distortionCoeffs,rvecs,tvecs);

    updateParameter("Camera matrix", K, Parameter::State);
    updateParameter("Distortion matrix", distortionCoeffs, Parameter::State);
    updateParameter("Reprojection error", quality, Parameter::State);
    lastCalibration = objectPointCollection.size();
}

void CalibrateStereoPair::Init(bool firstInit)
{
	updateParameter<boost::function<void(void)>>("Clear", boost::bind(&CalibrateStereoPair::clear, this));
    updateParameter("Num corners X", 6);
    updateParameter("Num corners Y", 9);
    updateParameter("Corner distance (mm)", double(18.75));
    updateParameter("Min pixel distance", 10.0);
    lastCalibration = 0;
}
void CalibrateStereoPair::clear()
{

}

cv::cuda::GpuMat CalibrateStereoPair::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateCamera)
NODE_DEFAULT_CONSTRUCTOR_IMPL(CalibrateStereoPair)
