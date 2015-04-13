#include "nodes/SerialStack.h"
using namespace EagleLib;

NODE_DEFAULT_CONSTRUCTOR_IMPL(SerialStack);

SerialStack::~SerialStack()
{

}
cv::cuda::GpuMat
SerialStack::doProcess(cv::cuda::GpuMat& img)
{
	
    for (auto it = children.begin(); it != children.end(); ++it)
	{
        img = getChild(it->id)->process(img);
    }
	return img;
}

bool SerialStack::SkipEmpty() const
{
    return false;
}
