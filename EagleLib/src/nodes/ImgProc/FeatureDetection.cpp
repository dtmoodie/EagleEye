#include "nodes/ImgProc/FeatureDetection.h"
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace EagleLib;

void GoodFeaturesToTrackDetector::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Feature Detector",
            cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1), Parameter::Output);
        updateParameter("Max corners",
            int(1000), Parameter::Control);
        updateParameter("Quality Level",
            double(0.01));
        updateParameter("Min Distance",
            double(0.0), Parameter::Control, "The minimum distance between detected points");
        updateParameter("Block Size",
            int(3));
        updateParameter("Use harris",
            false);
        updateParameter("Harris K",
            double(0.04));
        updateParameter("Enabled",
            false);
        updateParameter<boost::function<cv::cuda::GpuMat(cv::cuda::GpuMat, cv::cuda::Stream)>>("Detection functor",
            boost::bind(&GoodFeaturesToTrackDetector::detect, this, _1, _2));
    }
}


cv::cuda::GpuMat
GoodFeaturesToTrackDetector::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{

    if(parameters[1]->changed || parameters[2]->changed ||
       parameters[3]->changed || parameters[4]->changed ||
       parameters[5]->changed || parameters[6]->changed)
    {
        int numCorners = getParameter<int>(1)->data;
        double qualityLevel = getParameter<double>(2)->data;
        double minDistance = getParameter<double>(3)->data;
        int blockSize = getParameter<int>(4)->data;
        bool useHarris = getParameter<bool>(5)->data;
        double harrisK = getParameter<double>(6)->data;

        updateParameter(0,
            cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1,
                numCorners,qualityLevel,minDistance,blockSize,useHarris,harrisK));
        std::stringstream ss;
        ss << "Good features to track detector parameters updated: " << numCorners << " " << qualityLevel
           << " " << minDistance << " " << blockSize << " " << useHarris << " " << harrisK;
        log(Status, ss.str());
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
        parameters[4]->changed = false;
        parameters[5]->changed = false;
        parameters[6]->changed = false;
    }
    if(getParameter<bool>(7)->data)
        return detect(img);
    return img;
}

cv::cuda::GpuMat GoodFeaturesToTrackDetector::detect(cv::cuda::GpuMat img, cv::cuda::Stream& stream)
{

    if(img.channels() != 1)
    {
        // Internal greyscale conversion
        cv::cuda::cvtColor(img, greyImg, CV_BGR2GRAY,0, stream);
    }else
    {
        greyImg = img;
    }
    auto detectorParam = getParameter<cv::Ptr<cv::cuda::CornersDetector>>(0);
    if(detectorParam == nullptr)
    {
        return img;
    }
    cv::Ptr<cv::cuda::CornersDetector> detector = detectorParam->data;
    if(detector == nullptr)
    {
        log(Error, "Detector not built");
        return img;
    }
    detector->detect(greyImg, detectedCorners, cv::cuda::GpuMat(), stream);
    updateParameter("Detected Corners", detectedCorners, Parameter::Output);
    updateParameter("Num corners", detectedCorners.cols, Parameter::State);
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(GoodFeaturesToTrackDetector)
