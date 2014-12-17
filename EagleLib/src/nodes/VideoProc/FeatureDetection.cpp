#include "nodes/VideoProc/FeatureDetection.h"
#include <opencv2/cudafeatures2d.hpp>

using namespace EagleLib;

GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector():
    imgType(CV_8UC1)
{
    detector.reset(new OutputParameter<cv::Ptr<cv::cuda::CornersDetector> >("GoodFeaturesToTrackDetector", "Good features to track detector", cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1)));
    numCorners.reset(new InputParameter< int >("numCorners", " ", 1000));
    qualityLevel.reset(new InputParameter<double>("qualityLevel", " ", .01));
    minDistance.reset(new InputParameter<double>("minDistance",  " ", 0.0));
    blockSize.reset(new InputParameter<int>("blockSize", " ", 3));
    useHarris.reset(new InputParameter<bool>("useHarris", " ", true));
    harrisK.reset(new InputParameter<double>("harrisK", " ", .04));
    calculateFlag.reset(new InputParameter<bool>("calculateFlag", " ", .04));
    corners.reset(new OutputParameter<cv::cuda::GpuMat>("keyPoints", "Detected key points"));
    parameters.push_back(detector);
    parameters.push_back(numCorners);
    parameters.push_back(qualityLevel);
    parameters.push_back(minDistance);
    parameters.push_back(blockSize);
    parameters.push_back(useHarris);
    parameters.push_back(calculateFlag);
    parameters.push_back(corners);
}

cv::cuda::GpuMat
GoodFeaturesToTrackDetector::doProcess(cv::cuda::GpuMat& img)
{
    if(numCorners->changed || qualityLevel->changed || minDistance->changed || blockSize->changed)// || img.type() != imgType)
        detector.reset(new TypedParameter<cv::Ptr<cv::cuda::CornersDetector> >("GoodFeaturesToTrackDetector",
                                                                               "Good features to track detector",
                                                                               cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1,
                                                                                    numCorners->data, qualityLevel->data, minDistance->data,
                                                                                    blockSize->data,useHarris->data, harrisK->data)));
    cv::cuda::GpuMat grey = img;
    if(img.channels() != 1)
    {
        if(warningCallback)
            warningCallback("Img not greyscale, converting");
        cv::cuda::cvtColor(img,grey,cv::COLOR_BGR2GRAY);
    }
    if(calculateFlag->data)
        detector->data->detect(grey,corners->data);
    if(cpuCallback || gpuCallback || drawResults)
    {
        cv::Mat results(img), pts(corners->data);

    }
    return img;
}
