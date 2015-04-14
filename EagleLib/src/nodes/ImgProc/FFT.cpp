#include "nodes/ImgProc/FFT.h"
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaarithm")

void FFT::Init(bool firstInit)
{
    updateParameter("DFT rows flag", false);
    updateParameter("DFT scale flag", false);
    updateParameter("DFT inverse flag", false);
    updateParameter("DFT real output flag", false);
    updateParameter("Desired output", int(-1));
}

cv::cuda::GpuMat FFT::doProcess(cv::cuda::GpuMat &img)
{
    if(img.empty())
        return img;

    if(img.channels() > 2)
    {
        std::stringstream ss;
        ss << "Too many channels, can only handle 1 or 2 channel input. Input has ";
        ss << img.channels() << " channels.";
        log(Warning, ss.str());
        return img;
    }
    cv::cuda::GpuMat floatImg;
    if(img.depth() != CV_32F)
        img.convertTo(floatImg,CV_MAKETYPE(CV_32F,img.channels()));
    else
        floatImg = img;
    cv::cuda::GpuMat dest;
    int flags = 0;
    if(getParameter<bool>(0)->data)
        flags = flags | cv::DFT_ROWS;
    if(getParameter<bool>(1)->data)
        flags = flags | cv::DFT_SCALE;
    if(getParameter<bool>(2)->data)
        flags = flags | cv::DFT_INVERSE;
    if(getParameter<bool>(3)->data)
        flags = flags | cv::DFT_REAL_OUTPUT;
    cv::cuda::dft(floatImg,dest,img.size(),flags);
    if(int channel = getParameter<int>("Desired output")->data != -1)
    {
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(dest,channels);
        if(channel < channels.size())
            dest = channels[channel];
    }
    return dest;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(FFT);
