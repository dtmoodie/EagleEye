#include "nodes/ImgProc/Channels.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;

void ConvertToGrey::Init(bool firstInit)
{

}

cv::cuda::GpuMat ConvertToGrey::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat grey;
    try
    {
        cv::cuda::cvtColor(img, grey, cv::COLOR_BGR2GRAY, 0, stream);
    }catch(cv::Exception &err)
    {
        log(Error, err.what());
        return img;
    }

    return grey;
}

void ConvertToHSV::Init(bool firstInit)
{

}

cv::cuda::GpuMat ConvertToHSV::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
	return img;
}

void ExtractChannels::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Output Channel", int(0));
    }
    channelNum = getParameter<int>(0)->data;
    channelsBuffer.resize(5);
}

cv::cuda::GpuMat ExtractChannels::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    TIME
    std::vector<cv::cuda::GpuMat>* channels = channelsBuffer.getFront();
    TIME
    cv::cuda::split(img,*channels,stream);
    TIME
    for(size_t i = 0; i < channels->size(); ++i)
    {
        updateParameter("Channel " + std::to_string(i), (*channels)[i], Parameter::Output);
    }
    TIME
    if(parameters[0]->changed)
        channelNum = getParameter<int>(0)->data;
    TIME
    if(channelNum == -1)
        return img;
    if(channelNum < channels->size())
        return (*channels)[channelNum];
    else
        return (*channels)[0];
}
void ConvertDataType::Init(bool firstInit)
{
    if(firstInit)
    {
        EnumParameter dataType;
        dataType.addEnum(ENUM(CV_8U));
        dataType.addEnum(ENUM(CV_8S));
        dataType.addEnum(ENUM(CV_16U));
        dataType.addEnum(ENUM(CV_16S));
        dataType.addEnum(ENUM(CV_32S));
        dataType.addEnum(ENUM(CV_32F));
        dataType.addEnum(ENUM(CV_64F));
        updateParameter("Data type", dataType);
        updateParameter("Alpha", 255.0);
        updateParameter("Beta", 0.0);
    }
}

cv::cuda::GpuMat ConvertDataType::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat output;
    img.convertTo(output, getParameter<EnumParameter>(0)->data.currentSelection, getParameter<double>(1)->data, getParameter<double>(2)->data,stream);
    return output;
}

void Merge::Init(bool firstInit)
{
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Channel1");
        addInputParameter<cv::cuda::GpuMat>("Channel2");
        addInputParameter<cv::cuda::GpuMat>("Channel3");
    }
}

cv::cuda::GpuMat Merge::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    auto chan1 = getParameter<cv::cuda::GpuMat*>(0);
    auto chan2 = getParameter<cv::cuda::GpuMat*>(1);
    auto chan3 = getParameter<cv::cuda::GpuMat*>(2);
    std::vector<cv::cuda::GpuMat> channels;
    if(chan1->data)
        channels.push_back(*chan1->data);
    else
        channels.push_back(img);
    if(chan2->data)
        channels.push_back(*chan2->data);
    if(chan3->data)
        channels.push_back(*chan3->data);
    cv::cuda::merge(channels, mergedChannels,stream);
    return mergedChannels;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToGrey);
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToHSV);
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExtractChannels);
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertDataType);
NODE_DEFAULT_CONSTRUCTOR_IMPL(Merge);
