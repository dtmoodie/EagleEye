#include "nodes/VideoProc/OpticalFlow.h"
#include "nodes/VideoProc/Tracking.hpp"
#include <external_includes/cv_cudaarithm.hpp>
#include <external_includes/cv_cudaimgproc.hpp>
using namespace EagleLib;

#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cudaoptflow")
#endif
NODE_DEFAULT_CONSTRUCTOR_IMPL(SparsePyrLKOpticalFlow)
NODE_DEFAULT_CONSTRUCTOR_IMPL(BroxOpticalFlow)
NODE_DEFAULT_CONSTRUCTOR_IMPL(DensePyrLKOpticalFlow)

void DensePyrLKOpticalFlow::Init(bool firstInit)
{
	opt_flow = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(13, 13), 3, 30, false);
	updateParameter<int>("Window Size", 13);
	updateParameter<int>("Pyramid levels", 3);
	updateParameter<int>("Iterations", 30);
	updateParameter<bool>("Use initial flow", false);
}
cv::cuda::GpuMat DensePyrLKOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
	if (parameters[0]->changed || parameters[1]->changed || parameters[2]->changed || parameters[3]->changed)
	{
		int size = *getParameter<int>("Window Size")->Data();
		int levels = *getParameter<int>("Pyramid levels")->Data();
		int iters = *getParameter<int>("Iterations")->Data();
		opt_flow = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(size, size), levels, iters, false);
		parameters[0]->changed = false;
		parameters[1]->changed = false; 
		parameters[2]->changed = false;
		parameters[3]->changed = false;
	}


	if (img.channels() != 1)
	{
		cv::cuda::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY, 0, stream);
	}
	else
	{
		greyImg = img;
	}
	if (prevGreyImg.empty())
	{
		greyImg.copyTo(prevGreyImg, stream);
		return img;
	}
	opt_flow->calc(prevGreyImg, greyImg, flow, stream);
	//cv::Mat field(flow);
	//cv::Mat prev(prevGreyImg);
	//cv::Mat curr(greyImg);
	greyImg.copyTo(prevGreyImg, stream);
	updateParameter("FlowField", flow);
	return img;
}
void SparsePyrLKOpticalFlow::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Input Points");
        updateParameter("Window size", int(21));
        updateParameter("Max Levels", int(3));
        updateParameter("Iterations", int(30));
        updateParameter("Use initial flow", true);
        updateParameter("Enabled", false);
        parameters[1]->changed = true;
        addInputParameter<boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, std::string&, cv::cuda::Stream)>>("Display function");
    }
    /*updateParameter<boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::Stream)>>(
        "Set Reference", boost::bind(&SparsePyrLKOpticalFlow::setReferenceImage, this, _1, _2, _3), Parameters::Parameter::Output);

    updateParameter<TrackSparseFunctor>("Sparse Track Functor", boost::bind(&SparsePyrLKOpticalFlow::trackSparse, this, _1, _2, _3, _4, _5, _6, _7), Parameters::Parameter::Output);*/
}
void SparsePyrLKOpticalFlow::Serialize(ISimpleSerializer *pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(optFlow);
}

cv::cuda::GpuMat SparsePyrLKOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed ||
       parameters[4]->changed)
    {
		int winSize = *getParameter<int>(1)->Data();
		int levels = *getParameter<int>(2)->Data();
		int iters = *getParameter<int>(3)->Data();
		bool useInital = *getParameter<bool>(4)->Data();
        optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(winSize,winSize),levels,iters,useInital);
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
        parameters[4]->changed = false;
    }

    if(img.channels() != 1)
        cv::cuda::cvtColor(img,greyImg, cv::COLOR_BGR2GRAY,0, stream);
    else
        greyImg = img;
    if(refImg.empty())
    {
        refImg = greyImg;
        return img;
    }else
    {
		cv::cuda::GpuMat* inputPts = getParameter<cv::cuda::GpuMat>(0)->Data();
        if(!inputPts)
            return img;
        if(!inputPts->empty())
        {
            if(prevKeyPoints.size() != inputPts->size())
                inputPts->copyTo(prevKeyPoints, stream);
            cv::cuda::GpuMat status, error;
            trackSparse(refImg, greyImg,*inputPts, prevKeyPoints,status, error, stream);
        }
    }
    return img;
}
void SparsePyrLKOpticalFlow::setReferenceImage(cv::cuda::GpuMat img, cv::cuda::GpuMat keyPoints, cv::cuda::Stream& stream)
{
	refImg = img;
    refPts  = keyPoints;
    keyPoints.copyTo(trackedKeyPoints,stream);

}
void SparsePyrLKOpticalFlow::trackSparse(
        cv::cuda::GpuMat refImg, cv::cuda::GpuMat curImg,
        cv::cuda::GpuMat refPts, cv::cuda::GpuMat &trackedPts,
        cv::cuda::GpuMat& status, cv::cuda::GpuMat& err,
        cv::cuda::Stream stream)
{
    if(optFlow == nullptr)
    {
        //log(Error, "Optical flow not initialized correctly");
		NODE_LOG(error) << "Optical flow not initialized correctly";
        return;
    }
    if(trackedPts.empty())
    {
        trackedPts = cv::cuda::GpuMat(refPts.size(), refPts.type());
        refPts.copyTo(trackedPts,stream);
    }
    boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, std::string&, cv::cuda::Stream)>* display =
            getParameter<boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::GpuMat, std::string&, cv::cuda::Stream)>>("Display function")->Data();


    optFlow->calc(refImg, curImg, refPts, trackedPts, status, err, stream);
    if(display)
    {
        (*display)(curImg, refPts, trackedPts, status, fullTreeName, stream);
    }
    updateParameter("Tracked points", trackedPts, Parameters::Parameter::Output);
	updateParameter("Status", status, Parameters::Parameter::Output);
	updateParameter("Error", err, Parameters::Parameter::Output);

}

void BroxOpticalFlow::Init(bool firstInit)
{

}

cv::cuda::GpuMat BroxOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
	return img;
}


