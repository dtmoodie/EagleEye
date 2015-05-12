#include "nodes/Utility/DataHandling.h"

using namespace EagleLib;

void GetOutputImage::Init(bool firstInit)
{
    if(firstInit)
        addInputParameter<cv::cuda::GpuMat>("Input");
}

cv::cuda::GpuMat GetOutputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat* input = getParameter<cv::cuda::GpuMat*>("Input")->data;
    if(input == nullptr)
    {
        log(Status, "Input not defined");
        return img;
    }
    if(input->empty())
        log(Status, "Input is empty");
    return *input;
}
void ImageInfo::Init(bool firstInit)
{
    EnumParameter dataType;
    dataType.addEnum(ENUM(CV_8U));
    dataType.addEnum(ENUM(CV_8S));
    dataType.addEnum(ENUM(CV_16U));
    dataType.addEnum(ENUM(CV_16S));
    dataType.addEnum(ENUM(CV_32S));
    dataType.addEnum(ENUM(CV_32F));
    dataType.addEnum(ENUM(CV_64F));
    updateParameter<EnumParameter>("Type",dataType, Parameter::State);
}
cv::cuda::GpuMat ImageInfo::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    getParameter<EnumParameter>(0)->data.currentSelection = img.type();
    parameters[0]->changed = true;
    updateParameter<int>("Depth",img.depth(), Parameter::State);
    updateParameter<int>("Rows",img.rows, Parameter::State);
    updateParameter<int>("Cols", img.cols, Parameter::State);
    updateParameter<int>("Channels", img.channels(), Parameter::State);
    updateParameter<int>("Step", img.step, Parameter::State);
    updateParameter<int>("Ref count", *img.refcount, Parameter::State);
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(GetOutputImage)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageInfo)
