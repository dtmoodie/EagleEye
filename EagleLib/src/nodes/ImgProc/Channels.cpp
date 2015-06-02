#include "nodes/ImgProc/Channels.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace EagleLib;

void ConvertToGrey::Init(bool firstInit)
{
    Node::Init(firstInit);
}

cv::cuda::GpuMat ConvertToGrey::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat grey;
    try
    {
        TIME
        cv::cuda::cvtColor(img, grey, cv::COLOR_BGR2GRAY, 0, stream);
    }catch(cv::Exception &err)
    {
        log(Error, err.what());
        return img;
    }
    TIME
    return grey;
}

void ConvertToHSV::Init(bool firstInit)
{
    Node::Init(firstInit);
}

cv::cuda::GpuMat ConvertToHSV::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
	return img;
}

void ExtractChannels::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(firstInit)
    {
        updateParameter("Output Channel", int(0));
    }

    channelsBuffer.resize(5);
}

cv::cuda::GpuMat ExtractChannels::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    TIME
    channelNum = getParameter<int>(0)->data;
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
    Node::Init(firstInit);
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
    Node::Init(firstInit);
    if(firstInit)
    {
        addInputParameter<cv::cuda::GpuMat>("Channel1", "", MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
        addInputParameter<cv::cuda::GpuMat>("Channel2", "", MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
        addInputParameter<cv::cuda::GpuMat>("Channel3", "", MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
        addInputParameter<cv::cuda::GpuMat>("Channel4", "", MatQualifier<cv::cuda::GpuMat>::get(-1,-1,1));
    }
    qualifiersSet = false;
}

cv::cuda::GpuMat Merge::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    auto chan1 = getParameter<cv::cuda::GpuMat*>(0);
    auto chan2 = getParameter<cv::cuda::GpuMat*>(1);
    auto chan3 = getParameter<cv::cuda::GpuMat*>(2);
    auto chan4 = getParameter<cv::cuda::GpuMat*>(3);
//    if(qualifiersSet == false || chan1->changed)
//    {
//        int type = img.type();
//        int width = img.cols;
//        int height = img.rows;
//        boost::function<bool(const Parameter::Ptr&)> f;
//        if(chan1->changed)
//        {
//            if(chan1->data)
//            {
//                type = chan1->data->type();
//                width = chan1->data->cols;
//                height = chan1->data->rows;
//            }
//        }
//        f = GpuMatQualifier::get(width, height, 1, type);
//        updateInputQualifier<cv::cuda::GpuMat>(1,f);
//        updateInputQualifier<cv::cuda::GpuMat>(2,f);
//        updateInputQualifier<cv::cuda::GpuMat>(3,f);
//        parameters[0]->changed = false;
//        parameters[1]->changed = false;
//        parameters[2]->changed = false;
//        parameters[3]->changed = false;
//        qualifiersSet = true;
//    }
    std::vector<cv::cuda::GpuMat> channels;
    if(chan1->data)
        channels.push_back(*chan1->data);
    else
        channels.push_back(img);
    if(chan2->data)
        channels.push_back(*chan2->data);
    if(chan3->data)
        channels.push_back(*chan3->data);
    if(chan4->data)
        channels.push_back(*chan4->data);
    cv::cuda::merge(channels, mergedChannels,stream);
    return mergedChannels;
}
void Reshape::Init(bool firstInit)
{
    updateParameter("Channels", int(0));
    updateParameter("Rows", int(0));
}

cv::cuda::GpuMat Reshape::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    return img.reshape(getParameter<int>(0)->data, getParameter<int>(1)->data);
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToGrey)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToHSV)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExtractChannels)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertDataType)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Merge)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Reshape)
