#include "nodes/SerialStack.h"
using namespace EagleLib;

NODE_DEFAULT_CONSTRUCTOR_IMPL(SerialStack);

SerialStack::~SerialStack()
{

}
cv::cuda::GpuMat
SerialStack::doProcess(cv::cuda::GpuMat& img)
{
	

}

bool SerialStack::SkipEmpty() const
{
    return false;
}
