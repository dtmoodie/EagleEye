#include "Stereo.h"
#include "opencv2/cudastereo.hpp"

#if _WIN32
    #if _DEBUG
        RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300d.lib")
    #else
        RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300.lib")
    #endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudastereo")
#endif



using namespace EagleLib;

void StereoBM::Init(bool firstInit)
{

}

cv::cuda::GpuMat StereoBM::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}

void StereoBilateralFilter::Init(bool firstInit)
{

}

cv::cuda::GpuMat StereoBilateralFilter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}

void StereoBeliefPropagation::Init(bool firstInit)
{

}

cv::cuda::GpuMat StereoBeliefPropagation::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{

    return img;
}

void StereoConstantSpaceBP::Init(bool firstInit)
{

}

cv::cuda::GpuMat StereoConstantSpaceBP::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBM)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBilateralFilter)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoBeliefPropagation)
NODE_DEFAULT_CONSTRUCTOR_IMPL(StereoConstantSpaceBP)
