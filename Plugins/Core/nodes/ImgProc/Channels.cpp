#include "nodes/ImgProc/Channels.h"
#include <external_includes/cv_cudaimgproc.hpp>
#include <external_includes/cv_cudaarithm.hpp>
#include <Qualifiers.hpp>
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
		// log(Error, err.what());
		NODE_LOG(error) << err.what();
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
void ConvertColorspace::Init(bool firstInit)
{
	Parameters::EnumParameter param;
	param.addEnum(ENUM(cv::COLOR_BGR2BGRA));
	param.addEnum(ENUM(cv::COLOR_RGB2RGBA));
	param.addEnum(ENUM(cv::COLOR_BGRA2BGR));
	param.addEnum(ENUM(cv::COLOR_RGBA2RGB));
	param.addEnum(ENUM(cv::COLOR_BGR2RGBA));
	param.addEnum(ENUM(cv::COLOR_RGB2BGRA));
	param.addEnum(ENUM(cv::COLOR_RGBA2BGR));
	param.addEnum(ENUM(cv::COLOR_BGRA2RGB));
	param.addEnum(ENUM(cv::COLOR_BGR2RGB));
	param.addEnum(ENUM(cv::COLOR_RGB2BGR));
	param.addEnum(ENUM(cv::COLOR_BGRA2RGBA));
	param.addEnum(ENUM(cv::COLOR_RGBA2BGRA));
	param.addEnum(ENUM(cv::COLOR_BGR2GRAY));
	param.addEnum(ENUM(cv::COLOR_GRAY2BGR));
	param.addEnum(ENUM(cv::COLOR_GRAY2RGB));
	param.addEnum(ENUM(cv::COLOR_GRAY2BGRA));
	param.addEnum(ENUM(cv::COLOR_GRAY2RGBA));
	param.addEnum(ENUM(cv::COLOR_BGRA2GRAY));
	param.addEnum(ENUM(cv::COLOR_RGBA2GRAY));
	param.addEnum(ENUM(cv::COLOR_BGR2BGR565));
	param.addEnum(ENUM(cv::COLOR_RGB2BGR565));
	param.addEnum(ENUM(cv::COLOR_BGR5652BGR));
	param.addEnum(ENUM(cv::COLOR_BGR5652RGB));
	param.addEnum(ENUM(cv::COLOR_BGRA2BGR565));
	param.addEnum(ENUM(cv::COLOR_RGBA2BGR565));
	param.addEnum(ENUM(cv::COLOR_BGR5652BGRA));
	param.addEnum(ENUM(cv::COLOR_BGR5652RGBA));
	param.addEnum(ENUM(cv::COLOR_GRAY2BGR565));

	param.addEnum(ENUM(cv::COLOR_BGR5652GRAY));
	param.addEnum(ENUM(cv::COLOR_BGR2BGR555));
	param.addEnum(ENUM(cv::COLOR_RGB2BGR555));
	param.addEnum(ENUM(cv::COLOR_BGR5552BGR));
	param.addEnum(ENUM(cv::COLOR_BGR5552RGB));
	param.addEnum(ENUM(cv::COLOR_BGRA2BGR555));
	param.addEnum(ENUM(cv::COLOR_RGBA2BGR555));
	param.addEnum(ENUM(cv::COLOR_BGR5552BGRA));
	param.addEnum(ENUM(cv::COLOR_BGR5552RGBA));

	param.addEnum(ENUM(cv::COLOR_GRAY2BGR555));
	param.addEnum(ENUM(cv::COLOR_BGR5552GRAY));

	param.addEnum(ENUM(cv::COLOR_BGR2XYZ));
	param.addEnum(ENUM(cv::COLOR_RGB2XYZ));
	param.addEnum(ENUM(cv::COLOR_XYZ2BGR));
	param.addEnum(ENUM(cv::COLOR_XYZ2RGB));

	param.addEnum(ENUM(cv::COLOR_BGR2YCrCb));
	param.addEnum(ENUM(cv::COLOR_RGB2YCrCb));
	param.addEnum(ENUM(cv::COLOR_YCrCb2BGR));
	param.addEnum(ENUM(cv::COLOR_YCrCb2RGB));

	param.addEnum(ENUM(cv::COLOR_BGR2HSV));
	param.addEnum(ENUM(cv::COLOR_RGB2HSV));

	param.addEnum(ENUM(cv::COLOR_BGR2Lab));
	param.addEnum(ENUM(cv::COLOR_RGB2Lab));

	param.addEnum(ENUM(cv::COLOR_BGR2Luv));
	param.addEnum(ENUM(cv::COLOR_RGB2Luv));
	param.addEnum(ENUM(cv::COLOR_BGR2HLS));
	param.addEnum(ENUM(cv::COLOR_RGB2HLS));

	param.addEnum(ENUM(cv::COLOR_HSV2BGR));
	param.addEnum(ENUM(cv::COLOR_HSV2RGB));

	param.addEnum(ENUM(cv::COLOR_Lab2BGR));
	param.addEnum(ENUM(cv::COLOR_Lab2RGB));
	param.addEnum(ENUM(cv::COLOR_Luv2BGR));
	param.addEnum(ENUM(cv::COLOR_Luv2RGB));
	param.addEnum(ENUM(cv::COLOR_HLS2BGR));
	param.addEnum(ENUM(cv::COLOR_HLS2RGB));

	updateParameter("Conversion Code", param);
}
cv::cuda::GpuMat ConvertColorspace::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	auto buf =  resultBuffer.getFront();
	cv::cuda::cvtColor(img, buf->data, getParameter<Parameters::EnumParameter>(0)->Data()->getValue(), 0, stream);
	return buf->data;
}


void Magnitude::Init(bool firstInit)
{

}
cv::cuda::GpuMat Magnitude::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
	cv::cuda::magnitude(img, magnitude, stream);
	return magnitude; 
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
    channelNum = *getParameter<int>(0)->Data();
    std::vector<cv::cuda::GpuMat>* channels = channelsBuffer.getFront();
    TIME
    cv::cuda::split(img,*channels,stream);
    TIME
    for(size_t i = 0; i < channels->size(); ++i)
    {
        updateParameter("Channel " + std::to_string(i), (*channels)[i], Parameters::Parameter::Output);
    }
    TIME
    if(parameters[0]->changed)
        channelNum = *getParameter<int>(0)->Data();
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
		Parameters::EnumParameter dataType;
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
    img.convertTo(output, getParameter<Parameters::EnumParameter>(0)->Data()->currentSelection, *getParameter<double>(1)->Data(), *getParameter<double>(2)->Data(),stream);
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
    auto chan1 = getParameter<cv::cuda::GpuMat>(0);
    auto chan2 = getParameter<cv::cuda::GpuMat>(1);
    auto chan3 = getParameter<cv::cuda::GpuMat>(2);
    auto chan4 = getParameter<cv::cuda::GpuMat>(3);
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
    if(chan1->Data())
        channels.push_back(*chan1->Data());
    else
        channels.push_back(img);
    if(chan2->Data())
        channels.push_back(*chan2->Data());
    if(chan3->Data())
        channels.push_back(*chan3->Data());
    if(chan4->Data())
        channels.push_back(*chan4->Data());
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
    return img.reshape(*getParameter<int>(0)->Data(), *getParameter<int>(1)->Data());
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToGrey)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertToHSV)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertColorspace)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ExtractChannels)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ConvertDataType)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Merge)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Reshape)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Magnitude)

REGISTER_NODE_HIERARCHY(ConvertToGrey, Image, Processing)
REGISTER_NODE_HIERARCHY(ConvertToHSV, Image, Processing)
REGISTER_NODE_HIERARCHY(ConvertColorspace, Image, Processing)
REGISTER_NODE_HIERARCHY(ExtractChannels, Image, Processing)
REGISTER_NODE_HIERARCHY(ConvertDataType, Image, Processing)
REGISTER_NODE_HIERARCHY(Merge, Image, Processing)
REGISTER_NODE_HIERARCHY(Reshape, Image, Processing)
REGISTER_NODE_HIERARCHY(Magnitude, Image, Processing)
