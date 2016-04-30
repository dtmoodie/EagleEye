#include "nodes/ImgProc/Filters.h"

#include <parameters/ParameteredObjectImpl.hpp>

using namespace EagleLib;
using namespace EagleLib::Nodes;
void Sobel::Init(bool firstInit)
{

}

cv::cuda::GpuMat Sobel::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void Canny::Init(bool firstInit)	
{
    Node::Init(firstInit);
	updateParameter("Low thresh", 0.0);
	updateParameter("High thresh", 20.0);
	updateParameter("Aperature size", int(3));
	updateParameter("L2 Gradient", false);
	detector = cv::cuda::createCannyEdgeDetector(0, 20, 3, false);
}

cv::cuda::GpuMat Canny::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
	
	return img;
}

void Laplacian::Init(bool firstInit)
{
    Node::Init(firstInit);
}

cv::cuda::GpuMat Laplacian::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void BiLateral::Init(bool firstInit)
{
    Node::Init(firstInit);
}

cv::cuda::GpuMat BiLateral::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void MeanShiftFilter::Init(bool firstInit)
{
    Node::Init(firstInit);
}

cv::cuda::GpuMat MeanShiftFilter::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
return img;
}

void MeanShiftProc::Init(bool firstInit)
{
    Node::Init(firstInit);
}

cv::cuda::GpuMat MeanShiftProc::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}

void MeanShiftSegmentation::Init(bool firstInit)
{
    Node::Init(firstInit);
}

cv::cuda::GpuMat MeanShiftSegmentation::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(Sobel, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Canny, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(Laplacian, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(BiLateral, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(MeanShiftFilter, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(MeanShiftProc, Image, Processing)
NODE_DEFAULT_CONSTRUCTOR_IMPL(MeanShiftSegmentation, Image, Processing)


