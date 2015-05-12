#include "nodes/VideoProc/OpticalFlow.h"

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace EagleLib;

#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cuda -lopencv_cudaoptflow")
#endif
NODE_DEFAULT_CONSTRUCTOR_IMPL(SparsePyrLKOpticalFlow)
NODE_DEFAULT_CONSTRUCTOR_IMPL(BroxOpticalFlow)


void SparsePyrLKOpticalFlow::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter<boost::function<int(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::Stream)>>(
            "Set Reference", boost::bind(&SparsePyrLKOpticalFlow::setReferenceImage, this, _1, _2, _3), Parameter::Output);
        updateParameter("Window size", int(21));
        updateParameter("Max Levels", int(3));
        updateParameter("Iterations", int(30));
        updateParameter("Use initial flow", true);
        optFlow = cv::cuda::SparsePyrLKOpticalFlow::create();
    }
}

cv::cuda::GpuMat SparsePyrLKOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    if(parameters[0]->changed ||
       parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed)
    {
        int winSize = getParameter<int>(0)->data;
        int levels = getParameter<int>(1)->data;
        int iters = getParameter<int>(2)->data;
        bool useInital = getParameter<bool>(3)->data;
        optFlow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(winSize,winSize),levels,iters,useInital);
        parameters[0]->changed = false;
        parameters[1]->changed = false;
        parameters[2]->changed = false;
        parameters[3]->changed = false;
    }
    cv::cuda::GpuMat greyImg;
    if(img.channels() != 1)
        cv::cuda::cvtColor(img,greyImg, cv::COLOR_BGR2GRAY,0, stream);
    else
        greyImg = img;
    if(prevGreyImg.empty())
    {
        prevGreyImg = greyImg;
        return img;
    }else
    {
        optFlow->calc(prevGreyImg, greyImg, prevKeyPoints, trackedKeyPoints, status, error, stream);
        updateParameter("Tracked points", trackedKeyPoints);
        updateParameter("Status", status);
        updateParameter("Error", error);
    }
}
int SparsePyrLKOpticalFlow::setReferenceImage(cv::cuda::GpuMat img, cv::cuda::GpuMat keyPoints, cv::cuda::Stream stream, int idx)
{
    if(img.channels() != 1)
        cv::cuda::cvtColor(img, refImg, cv::COLOR_BGR2GRAY,0, stream);
    refPts  = keyPoints;
    keyPoints.copyTo(trackedKeyPoints,stream);

}

void BroxOpticalFlow::Init(bool firstInit)
{

}

cv::cuda::GpuMat BroxOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{

}


