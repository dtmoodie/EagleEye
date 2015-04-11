#include "nodes/ImgProc/Channels.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;

void ConvertToGrey::Init(bool firstInit)
{

}

cv::cuda::GpuMat ConvertToGrey::doProcess(cv::cuda::GpuMat &img)
{
    cv::cuda::GpuMat grey;
    cv::cuda::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    return grey;
}

void ConvertToHSV::Init(bool firstInit)
{

}

cv::cuda::GpuMat ConvertToHSV::doProcess(cv::cuda::GpuMat &img)
{

}

void ExtractChannels::Init(bool firstInit)
{
    updateParameter("Output Channel", int(0));
    channelNum = 0;
}

cv::cuda::GpuMat ExtractChannels::doProcess(cv::cuda::GpuMat &img)
{
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(img,channels);
    for(int i = 0; i < channels.size(); ++i)
    {
        updateParameter("Channel " + std::to_string(i), channels[i], Parameter::Output);
    }
    if(parameters[0]->changed)
        channelNum = getParameter<int>(0)->data;
    if(channelNum < channels.size())
        return channels[channelNum];
    else
        return channels[0];
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToGrey);
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToHSV);
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExtractChannels);
