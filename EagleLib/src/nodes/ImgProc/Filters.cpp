#include "nodes/ImgProc/Filters.h"

using namespace EagleLib;

void Sobel::Init(bool firstInit)
{

}

cv::cuda::GpuMat Sobel::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void Canny::Init(bool firstInit)
{

}

cv::cuda::GpuMat Canny::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void Laplacian::Init(bool firstInit)
{

}

cv::cuda::GpuMat Laplacian::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void BiLateral::Init(bool firstInit)
{

}

cv::cuda::GpuMat BiLateral::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void MeanShiftFilter::Init(bool firstInit)
{

}

cv::cuda::GpuMat MeanShiftFilter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void MeanShiftProc::Init(bool firstInit)
{

}

cv::cuda::GpuMat MeanShiftProc::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}

void MeanShiftSegmentation::Init(bool firstInit)
{

}

cv::cuda::GpuMat MeanShiftSegmentation::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(Sobel)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Canny)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Laplacian)
NODE_DEFAULT_CONSTRUCTOR_IMPL(BiLateral)
NODE_DEFAULT_CONSTRUCTOR_IMPL(MeanShiftFilter)
NODE_DEFAULT_CONSTRUCTOR_IMPL(MeanShiftProc)
NODE_DEFAULT_CONSTRUCTOR_IMPL(MeanShiftSegmentation)


