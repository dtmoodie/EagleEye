#include "nodes/VideoProc/OpticalFlow.h"

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
using namespace EagleLib;

#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#if __linux
    RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cuda -lopencv_cudaoptflow");
#endif

void SparsePyrLKOpticalFlow::Init(bool firstInit)
{
    updateParameter("Window size", int(21));
    updateParameter("Max Levels", int(3));
    updateParameter("Iterations", int(30));
    updateParameter("Use initial flow", true);
    addInputParameter<cv::cuda::GpuMat>("Key points");
    optFlow = cv::cuda::SparsePyrLKOpticalFlow::create();
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
    }

}

void BroxOpticalFlow::Init(bool firstInit)
{

}

cv::cuda::GpuMat BroxOpticalFlow::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{

}


