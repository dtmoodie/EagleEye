#include "nodes/Node.h"
#include "PCL_bridge.h"

RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core")
using namespace EagleLib;
IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}



cv::cuda::GpuMat PCL_bridge::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
    return img;
}
void PCL_bridge::Init(bool firstInit)
{

}


NODE_DEFAULT_CONSTRUCTOR_IMPL(PCL_bridge)
