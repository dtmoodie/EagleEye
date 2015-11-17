#include "nodes/SerialStack.h"
using namespace EagleLib;

NODE_DEFAULT_CONSTRUCTOR_IMPL(SerialStack);
REGISTER_NODE_HIERARCHY(SerialStack, Utility)

SerialStack::~SerialStack()
{

}
cv::cuda::GpuMat
SerialStack::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    return img;
}

bool SerialStack::SkipEmpty() const
{
    return false;
}
