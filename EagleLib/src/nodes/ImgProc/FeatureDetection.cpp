#include "nodes/ImgProc/FeatureDetection.h"
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace EagleLib;

void GoodFeaturesToTrackDetector::Init(bool firstInit)
{
    updateParameter("Feature Detector", cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1), Parameter::Output);
    updateParameter("Max corners", int(1000), Parameter::Control);
    updateParameter("Quality Level", double(0.01));
    updateParameter("Min Distance", double(0.0), Parameter::Control, "The minimum distance between detected points");
    updateParameter("Block Size", int(3));
    updateParameter("Use harris", false);
    updateParameter("Harris K", double(0.04));
    std::cout << "Initialization of good features to track detector" << std::endl;
}


cv::cuda::GpuMat
GoodFeaturesToTrackDetector::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream stream)
{
    cv::cuda::GpuMat greyImg;
    if(img.channels() != 1)
    {
        // Internal greyscale conversion
        cv::cuda::cvtColor(img, greyImg, CV_BGR2GRAY,0, stream);
    }else
    {
        greyImg = img;
    }
    auto detectorParam = getParameter<cv::Ptr<cv::cuda::CornersDetector>>("Feature Detector");
    if(detectorParam == nullptr)
        return img;
    cv::Ptr<cv::cuda::CornersDetector> detector = detectorParam->data;
    if(detector == nullptr)
        return img;
    cv::cuda::GpuMat detectedCorners;
    detector->detect(greyImg, detectedCorners, cv::cuda::GpuMat(), stream);
    updateParameter("Detected Corners", detectedCorners);
    return img;
}
NODE_DEFAULT_CONSTRUCTOR_IMPL(GoodFeaturesToTrackDetector)
