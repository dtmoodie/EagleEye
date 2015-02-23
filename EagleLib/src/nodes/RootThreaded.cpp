#include "nodes/RootThreaded.h"

using namespace EagleLib;


RootThreaded::RootThreaded()
{
    nodeName = "RootThreaded";
    //Node();
}
RootThreaded::~RootThreaded()
{

}
cv::cuda::GpuMat 
RootThreaded::doProcess(cv::cuda::GpuMat& img)
{
	return img;
}
