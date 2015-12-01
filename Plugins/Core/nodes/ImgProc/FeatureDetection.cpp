#include "nodes/ImgProc/FeatureDetection.h"
#include <external_includes/cv_cudafeatures2d.hpp>
#include <external_includes/cv_cudafilters.hpp>
#include <external_includes/cv_cudaoptflow.hpp>
#include <external_includes/cv_cudafeatures2d.hpp>
#include <external_includes/cv_cudaimgproc.hpp>
#include "nodes/VideoProc/Tracking.hpp"
using namespace EagleLib;

void GoodFeaturesToTrackDetector::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Feature Detector",
			cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1), Parameters::Parameter::Output);
        updateParameter("Max corners",
			int(1000), Parameters::Parameter::Control);
        updateParameter("Quality Level",
            double(0.01));
        updateParameter("Min Distance",
			double(0.0), Parameters::Parameter::Control, "The minimum distance between detected points");
        updateParameter("Block Size",
            int(3));
        updateParameter("Use harris",
            false);
        updateParameter("Harris K",
            double(0.04));
        updateParameter("Enabled",
            false);
        updateParameter<DetectAndComputeFunctor>("Detection functor",
			boost::bind(&GoodFeaturesToTrackDetector::detect, this, _1, _2, _3, _4, _5), Parameters::Parameter::Output);
        greyImgs.resize(5);
        addInputParameter<cv::cuda::GpuMat>("Mask");
    }
    detectedPoints.resize(5);
}


cv::cuda::GpuMat
GoodFeaturesToTrackDetector::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(parameters[1]->changed || parameters[2]->changed ||
       parameters[3]->changed || parameters[4]->changed ||
       parameters[5]->changed || parameters[6]->changed)
    {
        int numCorners = *getParameter<int>(1)->Data();
        double qualityLevel = *getParameter<double>(2)->Data();
        double minDistance = *getParameter<double>(3)->Data();
        int blockSize = *getParameter<int>(4)->Data();
        bool useHarris = *getParameter<bool>(5)->Data();
        double harrisK = *getParameter<double>(6)->Data();

        updateParameter(0,
            cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1,
            numCorners,qualityLevel,minDistance,blockSize,useHarris,harrisK));


		NODE_LOG(info) << "Good features to track detector parameters updated: " << numCorners << " " << qualityLevel
			<< " " << minDistance << " " << blockSize << " " << useHarris << " " << harrisK;

        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
        parameters[4]->changed = false;
        parameters[5]->changed = false;
        parameters[6]->changed = false;
    }
    if(!*getParameter<bool>(7)->Data())
        return img;
    cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat>("Mask")->Data();

    auto keyPoints = detectedPoints.getFront();
    if(mask)
    {
        detect(img, *mask, keyPoints->first, keyPoints->second, stream);
    }else
    {
        detect(img, cv::cuda::GpuMat(), keyPoints->first, keyPoints->second, stream);
    }
    return img;
}

void GoodFeaturesToTrackDetector::detect(cv::cuda::GpuMat img, cv::cuda::GpuMat mask,
                    cv::cuda::GpuMat& keyPoints,
                    cv::cuda::GpuMat& descriptors,
                    cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* greyImg = greyImgs.getFront();
    if(img.channels() != 1)
    {
        // Internal greyscale conversion
        cv::cuda::cvtColor(img, *greyImg, CV_BGR2GRAY,0, stream);
    }else
    {
        *greyImg = img;
    }
    auto detectorParam = getParameter<cv::Ptr<cv::cuda::CornersDetector>>(0);
    if(detectorParam == nullptr)
    {
        //log(Error, "Detector not built");
		NODE_LOG(error) << "Detector not built";
        return;
    }
    cv::Ptr<cv::cuda::CornersDetector> detector = *detectorParam->Data();
    if(detector == nullptr)
    {
        //log(Error, "Detector not built");
		NODE_LOG(error) << "Detector not built";
        return;
    }
    detector->detect(*greyImg, keyPoints, mask, stream);
	updateParameter("Detected Corners", keyPoints, Parameters::Parameter::Output);
	updateParameter("Num corners", keyPoints.cols, Parameters::Parameter::State);
}

/// *****************************************************************************************
/// *****************************************************************************************
/// *****************************************************************************************
/// // See if fast can use color images, probably not but worth a shot
void FastFeatureDetector::detect(cv::cuda::GpuMat img, cv::cuda::GpuMat mask,
            cv::cuda::GpuMat& keyPoints,
            cv::cuda::GpuMat& descriptors,
            cv::cuda::Stream& stream)
{

    cv::Ptr<cv::cuda::FastFeatureDetector> detector = *getParameter<cv::Ptr<cv::cuda::FastFeatureDetector>>(0)->Data();
    if(detector)
    {
        detector->detectAndComputeAsync(img,mask,keyPoints,descriptors,false, stream);
        updateParameter("KeyPoints", keyPoints);
        updateParameter("Descriptors", descriptors);
    }

}

void FastFeatureDetector::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Detector object", cv::cuda::FastFeatureDetector::create());
        updateParameter("Threshold", int(10));
        updateParameter("Use nonmax suppression", true);
		Parameters::EnumParameter param;
        param.addEnum(ENUM(cv::cuda::FastFeatureDetector::TYPE_5_8));
        param.addEnum(ENUM(cv::cuda::FastFeatureDetector::TYPE_7_12));
        param.addEnum(ENUM(cv::cuda::FastFeatureDetector::TYPE_9_16));
        param.currentSelection = 2;
        updateParameter("Type", param);
        updateParameter("Max detected points", int(5000));
        addInputParameter<cv::cuda::GpuMat>("Mask");
    }
    detectedPoints.resize(5);
    updateParameter<DetectAndComputeFunctor>("Detection Functor", boost::bind(&FastFeatureDetector::detect, this, _1, _2, _3, _4, _5));
}

cv::cuda::GpuMat FastFeatureDetector::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed ||
       parameters[4]->changed)
    {
        updateParameter(0, cv::cuda::FastFeatureDetector::create(
                            *getParameter<int>(1)->Data(),
                            *getParameter<bool>(2)->Data(),
                            getParameter<Parameters::EnumParameter>(3)->Data()->getValue(),
                            *getParameter<int>(4)->Data()));
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
        parameters[4]->changed = false;
    }
    cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat>("Mask")->Data();
    auto keyPoints = detectedPoints.getFront();
    if(mask)
    {
        detect(img, *mask, keyPoints->first, keyPoints->second, stream);
    }else
    {
        detect(img, cv::cuda::GpuMat(), keyPoints->first, keyPoints->second, stream);
    }
    return img;
}
/// *****************************************************************************************
/// *****************************************************************************************
/// *****************************************************************************************
void ORBFeatureDetector::detect(cv::cuda::GpuMat img, cv::cuda::GpuMat mask,
            cv::cuda::GpuMat& keyPoints,
            cv::cuda::GpuMat& descriptors,
            cv::cuda::Stream& stream)
{
    cv::Ptr<cv::cuda::ORB> detector = *getParameter<cv::Ptr<cv::cuda::ORB>>(0)->Data();
    if(detector)
    {
        detector->detectAndComputeAsync(img,mask,keyPoints,descriptors,false, stream);
    }
}

void ORBFeatureDetector::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Detector", cv::cuda::ORB::create(500,1.5,8,31,0,2,cv::ORB::HARRIS_SCORE, 31,20,true));
        updateParameter("Number of features", int(500));    //1
        updateParameter("Scale Factor", float(1.2));        //2
        updateParameter("Num Levels", int(8));              //3
        updateParameter("Edge Threshold", int(31));         //4
        updateParameter("First Level", int(0));             //5
        updateParameter("WTA_K", int(2));                   //6
		Parameters::EnumParameter param;
        param.addEnum(ENUM(cv::ORB::kBytes));
        param.addEnum(ENUM(cv::ORB::HARRIS_SCORE));
        param.addEnum(ENUM(cv::ORB::FAST_SCORE));
        param.currentSelection = 1;
        updateParameter("Score Type", param);               //7
        updateParameter("Patch Size", int(31));             //8
        updateParameter("Fast Threshold", int(20));         //9
        updateParameter("Blur for Descriptors", true);      //10
        addInputParameter<cv::cuda::GpuMat>("Mask");
    }
    updateParameter<DetectAndComputeFunctor>("Detection Functor", boost::bind(&ORBFeatureDetector::detect, this, _1, _2, _3, _4, _5));

}

cv::cuda::GpuMat ORBFeatureDetector::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed ||
       parameters[4]->changed ||
       parameters[5]->changed ||
       parameters[6]->changed ||
       parameters[7]->changed ||
       parameters[8]->changed ||
       parameters[9]->changed ||
       parameters[10]->changed)
    {
        updateParameter(0,
            cv::cuda::ORB::create(
			*getParameter<int>(1)->Data(),
					*getParameter<float>(2)->Data(),
					*getParameter<int>(3)->Data(),
					*getParameter<int>(4)->Data(),
					*getParameter<int>(5)->Data(),
					*getParameter<int>(6)->Data(),
					getParameter<Parameters::EnumParameter>(7)->Data()->getValue(),
					*getParameter<int>(8)->Data(),
					*getParameter<int>(9)->Data(),
					*getParameter<bool>(10)->Data()));

       parameters[1]->changed = false;
       parameters[2]->changed = false;
       parameters[3]->changed = false;
       parameters[4]->changed = false;
       parameters[5]->changed = false;
       parameters[6]->changed = false;
       parameters[7]->changed = false;
       parameters[8]->changed = false;
       parameters[9]->changed = false;
       parameters[10]->changed = false;
    }
	cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat>("Mask")->Data();
    auto keyPoints = detectedPoints.getFront();
    if(mask)
    {
        detect(img, *mask, keyPoints->first, keyPoints->second, stream);
    }else
    {
        detect(img, cv::cuda::GpuMat(), keyPoints->first, keyPoints->second, stream);
    }
    return img;
}
void HistogramRange::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(levels)
}

void HistogramRange::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter<double>("Lower bound", 0.0);
        updateParameter<double>("Upper bound", 1.0);
        updateParameter<int>("Bins", 100);
        updateParameter<cv::cuda::GpuMat>("Histogram", cv::cuda::GpuMat(), Parameters::Parameter::Output);
        updateLevels(CV_8U);
    }
}

cv::cuda::GpuMat HistogramRange::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed ||
       parameters[1]->changed ||
       parameters[2]->changed)
    {
        updateLevels(img.type());
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
    }
    if(img.channels() == 1 || img.channels() == 4)
    {
        TIME
        cv::cuda::GpuMat hist;
        cv::cuda::histRange(img, hist, levels, stream);
        TIME
        updateParameter(3, hist);
        TIME
        if(parameters[3]->subscribers > 0)
            return img;
        return hist;
    }else
    {
//        log(Warning, "Multi channel histograms not supported for " + boost::lexical_cast<std::string>(img.channels()) + " channels");
		NODE_LOG(warning) << "Multi channel histograms not supported for " + boost::lexical_cast<std::string>(img.channels()) + " channels";
    }
    return img;
}

void HistogramRange::updateLevels(int type)
{
	double lower = *getParameter<double>(0)->Data();
	double upper = *getParameter<double>(1)->Data();
	int bins = *getParameter<int>(2)->Data();
    cv::Mat h_mat;
    if(type == CV_32F)
        h_mat = cv::Mat(1, bins, CV_32F);
    else
        h_mat = cv::Mat(1, bins, CV_32S);
    double step = (upper - lower) / double(bins);

    double val = lower;
    for(int i = 0; i < bins; ++i, val += step)
    {
        if(type == CV_32F)
            h_mat.at<float>(i) = val;
        if(type == CV_8U)
            h_mat.at<int>(i) = val;
    }
    levels.upload(h_mat);
    updateParameter("Histogram bins", h_mat, Parameters::Parameter::Output);
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(GoodFeaturesToTrackDetector)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ORBFeatureDetector)
NODE_DEFAULT_CONSTRUCTOR_IMPL(FastFeatureDetector)
NODE_DEFAULT_CONSTRUCTOR_IMPL(HistogramRange)

REGISTER_NODE_HIERARCHY(GoodFeaturesToTrackDetector, Image, Extractor)
REGISTER_NODE_HIERARCHY(ORBFeatureDetector, Image, Extractor)
REGISTER_NODE_HIERARCHY(FastFeatureDetector, Image, Extractor)
REGISTER_NODE_HIERARCHY(HistogramRange, Image, Extractor)