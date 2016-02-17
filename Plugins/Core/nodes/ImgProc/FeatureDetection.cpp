#include "nodes/ImgProc/FeatureDetection.h"

#include <EagleLib/rcc/external_includes/cv_cudafilters.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaoptflow.hpp>
#include <EagleLib/rcc/external_includes/cv_cudaimgproc.hpp>
#include "EagleLib/nodes/VideoProc/Tracking.hpp"
#include "EagleLib/utilities/GpuMatAllocators.h"
#include <EagleLib/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;

void GoodFeaturesToTrackDetector::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Max corners",			int(1000));
        updateParameter("Quality Level",            double(0.01));
        updateParameter("Min Distance",			double(0.0))->SetTooltip("The minimum distance between detected points");
        updateParameter("Block Size",            int(3));
        updateParameter("Use harris",            false);
        updateParameter("Harris K",            double(0.04));
        
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
       parameters[5]->changed || parameters[6]->changed || detector == nullptr)
    {

        int numCorners = *getParameter<int>(0)->Data();
        double qualityLevel = *getParameter<double>(1)->Data();
        double minDistance = *getParameter<double>(2)->Data();
        int blockSize = *getParameter<int>(3)->Data();
        bool useHarris = *getParameter<bool>(4)->Data();
        double harrisK = *getParameter<double>(5)->Data();

        detector = cv::cuda::createGoodFeaturesToTrackDetector(img.type(),
            numCorners, qualityLevel, minDistance, blockSize, useHarris, harrisK);


		NODE_LOG(info) << "Good features to track detector parameters updated: " << numCorners << " " << qualityLevel
			<< " " << minDistance << " " << blockSize << " " << useHarris << " " << harrisK;

        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
        parameters[4]->changed = false;
        parameters[5]->changed = false;
    }

    cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat>("Mask")->Data();

    cv::cuda::GpuMat grey_img;
    if(img.channels() != 1)
    {
        cv::cuda::cvtColor(img, grey_img, cv::COLOR_BGR2GRAY, 0, stream);

    }else
    {
        grey_img = img;
    }
    cv::cuda::GpuMat key_points;
    if(mask)
    {

        detector->detect(grey_img, key_points, *mask, stream);
    }else
    {
        detector->detect(grey_img, key_points, cv::cuda::GpuMat(), stream);
    }
    updateParameter("Detected Corners", key_points)->type =  Parameters::Parameter::Output;
    updateParameter("Num corners", key_points.cols)->type = Parameters::Parameter::State;
    return img;
}

void FastFeatureDetector::Init(bool firstInit)
{
    if(firstInit)
    {
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
}
void FastFeatureDetector::Serialize(ISimpleSerializer *pSerializer)
{
	Node::Serialize(pSerializer);
	SERIALIZE(detector);
}
cv::cuda::GpuMat FastFeatureDetector::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed ||
       parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed)
    {
		detector = cv::cuda::FastFeatureDetector::create(
			*getParameter<int>(1)->Data(),
			*getParameter<bool>(2)->Data(),
			getParameter<Parameters::EnumParameter>(3)->Data()->getValue(),
			*getParameter<int>(4)->Data());
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
    }
    cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat>("Mask")->Data();
	cv::cuda::GpuMat key_points(BlockMemoryAllocator::Instance());
    if(mask)
    {
		detector->detectAsync(img, key_points, *mask, stream);
    }else
    {
		detector->detectAsync(img, key_points, cv::cuda::GpuMat(), stream);
    }
	if (!key_points.empty())
		updateParameter("Detected Key Points", key_points);
    return img;
}
/// *****************************************************************************************
/// *****************************************************************************************
/// *****************************************************************************************


void ORBFeatureDetector::Init(bool firstInit)
{
    if(firstInit)
    {
		detector = cv::cuda::ORB::create();
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
    
}
void ORBFeatureDetector::Serialize(ISimpleSerializer* pSerializer)
{
	Node::Serialize(pSerializer);
	SERIALIZE(detector);
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
       parameters[0]->changed || detector == nullptr)
    {
        detector = cv::cuda::ORB::create(
					*getParameter<int>(0)->Data(),
					*getParameter<float>(1)->Data(),
					*getParameter<int>(2)->Data(),
					*getParameter<int>(3)->Data(),
					*getParameter<int>(4)->Data(),
					*getParameter<int>(5)->Data(),
					getParameter<Parameters::EnumParameter>(6)->Data()->getValue(),
					*getParameter<int>(7)->Data(),
					*getParameter<int>(8)->Data(),
					*getParameter<bool>(9)->Data());

       parameters[1]->changed = false;
       parameters[2]->changed = false;
       parameters[3]->changed = false;
       parameters[4]->changed = false;
       parameters[5]->changed = false;
       parameters[6]->changed = false;
       parameters[7]->changed = false;
       parameters[8]->changed = false;
       parameters[9]->changed = false;
       parameters[0]->changed = false;
    }
	cv::cuda::GpuMat* mask = getParameter<cv::cuda::GpuMat>("Mask")->Data();
	
	cv::cuda::GpuMat key_points(BlockMemoryAllocator::Instance()), point_descriptors(BlockMemoryAllocator::Instance());
	if (detector == nullptr)
		return img;
    if(mask)
    {
		detector->detectAndComputeAsync(img, *mask, key_points, point_descriptors, false, stream);
    }else
    {
		detector->detectAndComputeAsync(img, cv::cuda::GpuMat(), key_points, point_descriptors, false, stream);
    }
	updateParameter("Detected Points", key_points);
	updateParameter("Point Descriptors", point_descriptors);
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
        updateParameter<cv::cuda::GpuMat>("Histogram", cv::cuda::GpuMat())->type =  Parameters::Parameter::Output;
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
        updateParameter(3, hist, &stream);
        TIME
        if(parameters[3]->subscribers > 0)
            return img;
        return hist;
    }else
    {
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
    updateParameter("Histogram bins", h_mat)->type =  Parameters::Parameter::Output;
}
void CornerHarris::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        updateParameter("Block size", 3);
        updateParameter("Sobel aperature size", 5);
        updateParameter("Harris free parameter", 1.0);
    }
    
}
cv::cuda::GpuMat CornerHarris::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed || parameters[1]->changed || parameters[2]->changed || detector == nullptr)
    {
        detector = cv::cuda::createHarrisCorner(img.type(),*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data(), *getParameter<double>(2)->Data());
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
    }
    cv::cuda::GpuMat score;
    detector->compute(img, score, stream);
    updateParameter("Corner score", score, &stream)->type = Parameters::Parameter::Output;
    return img;
}
void CornerMinEigenValue::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        updateParameter("Block size", 3);
        updateParameter("Sobel aperature size", 5);
        updateParameter("Harris free parameter", 1.0);
    }
    
}
cv::cuda::GpuMat CornerMinEigenValue::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if (parameters[0]->changed || parameters[1]->changed || detector == nullptr)
    {
        detector = cv::cuda::createMinEigenValCorner(img.type(), *getParameter<int>(0)->Data(), *getParameter<int>(1)->Data());
    }
    cv::cuda::GpuMat score;
    detector->compute(img, score, stream);
    updateParameter("Corner score", score, &stream)->type = Parameters::Parameter::Output;
    return img;
}



NODE_DEFAULT_CONSTRUCTOR_IMPL(GoodFeaturesToTrackDetector, Image, Extractor);
NODE_DEFAULT_CONSTRUCTOR_IMPL(ORBFeatureDetector, Image, Extractor);
NODE_DEFAULT_CONSTRUCTOR_IMPL(FastFeatureDetector, Image, Extractor);
NODE_DEFAULT_CONSTRUCTOR_IMPL(HistogramRange, Image, Extractor);
NODE_DEFAULT_CONSTRUCTOR_IMPL(CornerHarris, Image, Extractor);


