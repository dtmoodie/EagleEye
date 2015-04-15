#include "nodes/Utility/DataHandling.h"

using namespace EagleLib;

void GetOutputImage::Init(bool firstInit)
{
    addInputParameter<cv::cuda::GpuMat>("Input");
}

cv::cuda::GpuMat GetOutputImage::doProcess(cv::cuda::GpuMat &img)
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



NODE_DEFAULT_CONSTRUCTOR_IMPL(GetOutputImage)
