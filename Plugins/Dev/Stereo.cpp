#include "Stereo.h"


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
    if(firstInit)
    {
        updateParameter("Num disparities", int(64));
        updateParameter("Block size", int(19));
        addInputParameter<cv::cuda::GpuMat>("Left image");
        addInputParameter<cv::cuda::GpuMat>("Right image");
    }
    stereoBM = cv::cuda::createStereoBM(*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data());
}

cv::cuda::GpuMat StereoBM::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    if(parameters[0]->changed || parameters[1]->changed)
    {
        stereoBM = cv::cuda::createStereoBM(*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data());
    }
    cv::cuda::GpuMat* left = *getParameter<cv::cuda::GpuMat*>(2)->Data();
    cv::cuda::GpuMat* right = *getParameter<cv::cuda::GpuMat*>(3)->Data();
    if(left == nullptr)
    {
        left = &img;
    }
    if(right == nullptr)
    {
        log(Error, "No input selected for right image");
        return img;
    }
    if(left->size() != right->size())
    {
        log(Error, "Images are of mismatched size");
        return img;
    }
    if(left->channels() != right->channels())
    {
        log(Error, "Images are of mismatched channels");
        return img;
    }
    auto buf = disparityBuf.getFront();

    stereoBM->compute(*left,*right,buf->data, stream);
    buf->record(stream);
    return buf->data;
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
