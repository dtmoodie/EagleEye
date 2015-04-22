#include "nodes/Utility/DataHandling.h"

using namespace EagleLib;

void GetOutputImage::Init(bool firstInit)
{
    if(firstInit)
        addInputParameter<cv::cuda::GpuMat>("Input");
}

cv::cuda::GpuMat GetOutputImage::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
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

}
cv::cuda::GpuMat ImageInfo::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    updateParameter<int>("Type",img.type());
    updateParameter<int>("Depth",img.depth());
    updateParameter<int>("Rows",img.rows);
    updateParameter<int>("Cols", img.cols);
    updateParameter<int>("Channels", img.channels());
    updateParameter<int>("Step", img.step);
    updateParameter<int>("Ref count", *img.refcount);
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(GetOutputImage)
NODE_DEFAULT_CONSTRUCTOR_IMPL(ImageInfo)
