#include "BundleAdjustment.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;

void BundleAdjustment::Init(bool firstInit)
{
}

cv::cuda::GpuMat BundleAdjustment::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(BundleAdjustment, Image);