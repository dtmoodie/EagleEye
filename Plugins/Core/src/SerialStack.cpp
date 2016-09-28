#include "nodes/SerialStack.h"

#include <parameters/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;

NODE_DEFAULT_CONSTRUCTOR_IMPL(SerialStack, Utility);


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
