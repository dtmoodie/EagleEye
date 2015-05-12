#include "nodes/VideoProc/OpticalFlow.h"
#include "nodes/VideoProc/Tracking.hpp"
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace EagleLib;

#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cudaoptflow")
#endif
NODE_DEFAULT_CONSTRUCTOR_IMPL(SparsePyrLKOpticalFlow)
NODE_DEFAULT_CONSTRUCTOR_IMPL(BroxOpticalFlow)


void SparsePyrLKOpticalFlow::Init(bool firstInit)
{
    if(firstInit)
    {
<<<<<<< HEAD
        updateParameter<boost::function<int(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::Stream)>>(
            "Set Reference", boost::bind(&SparsePyrLKOpticalFlow::setReferenceImage, this, _1, _2, _3), Parameter::Output);
=======
        addInputParameter<cv::cuda::GpuMat>("Input Points");
>>>>>>> 591e3beeebd8738622ec58f3a2913592780a1ecd
        updateParameter("Window size", int(21));
        updateParameter("Max Levels", int(3));
        updateParameter("Iterations", int(30));
        updateParameter("Use initial flow", true);
        updateParameter("Enabled", false);
        parameters[1]->changed = true;
    }
    updateParameter<boost::function<void(cv::cuda::GpuMat, cv::cuda::GpuMat, cv::cuda::Stream)>>(
        "Set Reference", boost::bind(&SparsePyrLKOpticalFlow::setReferenceImage, this, _1, _2, _3), Parameter::Output);
    updateParameter<TrackSparseFunctor>("Sparse Track Functor", boost::bind(&SparsePyrLKOpticalFlow::trackSparse, this, _1,_2,_3, _4, _5, _6, _7));
}

cv::cuda::GpuMat SparsePyrLKOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(parameters[1]->changed ||
       parameters[2]->changed ||
       parameters[3]->changed ||
       parameters[4]->changed)
    {
        int winSize = getParameter<int>(1)->data;
        int levels = getParameter<int>(2)->data;
        int iters = getParameter<int>(3)->data;
        bool useInital = getParameter<bool>(4)->data;
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
        cv::cuda::GpuMat* inputPts = getParameter<cv::cuda::GpuMat*>(0)->data;
        if(!inputPts)
            return img;
        if(!inputPts->empty())
        {
            if(prevKeyPoints.size() != inputPts->size())
                inputPts->copyTo(prevKeyPoints, stream);
            cv::cuda::GpuMat status, error;
            prevKeyPoints = trackSparse(refImg, greyImg,*inputPts, prevKeyPoints,status, error, stream);
        }
    }
    return img;
}
<<<<<<< HEAD
int SparsePyrLKOpticalFlow::setReferenceImage(cv::cuda::GpuMat img, cv::cuda::GpuMat keyPoints, cv::cuda::Stream stream, int idx)
=======
void SparsePyrLKOpticalFlow::setReferenceImage(cv::cuda::GpuMat img, cv::cuda::GpuMat keyPoints, cv::cuda::Stream& stream)
>>>>>>> 591e3beeebd8738622ec58f3a2913592780a1ecd
{
    if(img.channels() != 1)
        cv::cuda::cvtColor(img, refImg, cv::COLOR_BGR2GRAY,0, stream);
    refPts  = keyPoints;
    keyPoints.copyTo(trackedKeyPoints,stream);

}
cv::cuda::GpuMat SparsePyrLKOpticalFlow::trackSparse(cv::cuda::GpuMat refImg, cv::cuda::GpuMat curImg,
    cv::cuda::GpuMat refPts, cv::cuda::GpuMat prevPts, cv::cuda::GpuMat& status, cv::cuda::GpuMat& err,
    cv::cuda::Stream stream)
{
    if(optFlow == nullptr)
    {
        log(Error, "Optical flow not initialized correctly");
        return prevPts;
    }
    optFlow->calc(refImg, curImg, refPts, prevPts, status, err, stream);
    updateParameter("Tracked points", prevPts, Parameter::Output);
    updateParameter("Status", status, Parameter::Output);
    updateParameter("Error", err, Parameter::Output);
    return prevPts;
}

void BroxOpticalFlow::Init(bool firstInit)
{

}

cv::cuda::GpuMat BroxOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

}


