#include "nodes/ImgProc/FeatureDetection.h"
#include <opencv2/cudafeatures2d.hpp>

using namespace EagleLib::Features2D;

GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector():
    imgType(CV_8UC1)
{
    addParameter("goodFeaturesToTrackDetector",cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1),"Good features to track detector", Parameter::Output);
    addParameter("numCorners", int(1000), "Number of corners to try to detect");
    addParameter("qualityLevel", 0.01, "Min relative quality level to keep. IE best corner scores 1500, qualityLevel=0.01 means rejection of anything below 15");
    addParameter("minDistance", 0.0, "Minimum distance between points");
    addParameter("blockSize", int(3), "Corner detection block search size");
    addParameter("useHarris", true, "Use harris corner detector");
    addParameter("harrisK", 0.04, "Harris corner detector free parameter");
    addParameter("calculateFlag", true, "Set flag to false to disable calculation");
    addParameter("keyPoints", cv::cuda::GpuMat(), "Detected key points", Parameter::Output);
    nodeName = std::string("GoodFeaturesToTrackDetector");
}
GoodFeaturesToTrackDetector::GoodFeaturesToTrackDetector(bool drawResults_):
    GoodFeaturesToTrackDetector()
{
    drawResults = drawResults_;
}

cv::cuda::GpuMat
GoodFeaturesToTrackDetector::doProcess(cv::cuda::GpuMat& img)
{
    auto detector       = getParameter<cv::Ptr<cv::cuda::CornersDetector> >(0);
    auto numCorners     = getParameter<int>(1);
    auto qualityLevel   = getParameter<double>(2);
    auto minDistance    = getParameter<double>(3);
    auto blockSize      = getParameter<int>(4);
    auto useHarris      = getParameter<bool>(5);
    auto harrisK        = getParameter<double>(6);
    auto calculateFlag  = getParameter<bool>(7);
    auto corners        = getParameter<cv::cuda::GpuMat>(8);

    //boost::shared_ptr< TypedParameter< cv::Ptr<cv::cuda::CornersDetector> > > detector = getParameter<cv::Ptr<cv::cuda::CornersDetector> >(0);
    //boost::shared_ptr< TypedParameter< int > > numCorners = getParameter<int>(1);
    //boost::shared_ptr< TypedParameter< double > > qualityLevel = getParameter<double>(2);
    //boost::shared_ptr< TypedParameter< double > > minDistance = getParameter<double>(3);
    //boost::shared_ptr< TypedParameter< int > > blockSize = getParameter<int>(4);
    //boost::shared_ptr< TypedParameter< bool > > useHarris = getParameter<bool>(5);
    //boost::shared_ptr< TypedParameter< double> > harrisK = getParameter<double>(6);
    //boost::shared_ptr< TypedParameter< bool > > calculateFlag = getParameter<bool>(7);
    //boost::shared_ptr< TypedParameter< cv::cuda::GpuMat> >corners = getParameter<cv::cuda::GpuMat>(8);



    if(numCorners->changed || qualityLevel->changed || minDistance->changed || blockSize->changed)
        parameters[0].reset(new TypedParameter<cv::Ptr<cv::cuda::CornersDetector> >("goodFeaturesToTrackDetector",
                                                                               "Good features to track detector",
                                                                               cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1,
                                                                                    numCorners->data, qualityLevel->data, minDistance->data,
                                                                                    blockSize->data,useHarris->data, harrisK->data),
                                                                               Parameter::Output));

    cv::cuda::GpuMat grey;
    if(img.channels() != 1)
    {
        if(warningCallback)
            warningCallback("Img not greyscale, converting");
        cv::cuda::cvtColor(img,grey,cv::COLOR_BGR2GRAY);
    }else
        grey = img;
    if(calculateFlag->data)
        detector->data->detect(grey,corners->data);
    if(cpuDisplayCallback || gpuDisplayCallback || drawResults)
    {
        cv::Mat results(img), pts(corners->data);
        if(!results.empty() && !pts.empty())
        {
            for(int i = 0; i < pts.cols; ++i)
            {
                cv::Point2f pt = pts.at<cv::Point2f>(i);
                cv::circle(results,pt, 10, cv::Scalar(0,0,255), 2);
            }
        }
        if(drawResults)
            img.upload(results);
        if(cpuDisplayCallback)
            cpuDisplayCallback(results);
        if(gpuDisplayCallback)
            gpuDisplayCallback(cv::cuda::GpuMat(results));
    }
    return img;
}
