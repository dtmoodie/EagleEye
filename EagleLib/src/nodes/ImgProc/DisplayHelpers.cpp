#include "nodes/ImgProc/DisplayHelpers.h"
using namespace EagleLib;
#include <opencv2/cudaarithm.hpp>


#ifdef _MSC_VER

#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaarithm")
#endif

void
AutoScale::Init(bool firstInit)
{

}

cv::cuda::GpuMat
AutoScale::doProcess(cv::cuda::GpuMat &img)
{
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(img,channels);
    for(int i = 0; i < channels.size(); ++i)
    {
        double minVal, maxVal;
        cv::cuda::minMax(channels[i], &minVal, &maxVal);
        double scaleFactor = 255.0 / (maxVal - minVal);
        channels[i].convertTo(channels[0], CV_8U, scaleFactor, minVal*scaleFactor);
    }


    cv::cuda::merge(channels,img);
    return img;
}

void
Colormap::Init(bool firstInit)
{

}

cv::cuda::GpuMat
Colormap::doProcess(cv::cuda::GpuMat &img)
{

}
NODE_DEFAULT_CONSTRUCTOR_IMPL(AutoScale);
NODE_DEFAULT_CONSTRUCTOR_IMPL(Colormap);
