#include "nodes/Utility/Cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

RUNTIME_COMPILER_LINKLIBRARY("-lcudart")
using namespace EagleLib;

void SetDevice::Init(bool firstInit)
{

    updateParameter<unsigned int>("Device Number", 0);
    updateParameter<std::string>("Device name","");
    firstRun = true;
}

cv::cuda::GpuMat SetDevice::doProcess(cv::cuda::GpuMat &img)
{
    int currentDevice = cv::cuda::getDevice();
    int maxDevice = cv::cuda::getCudaEnabledDeviceCount();
    if(firstRun)
    {
        std::stringstream ss;
        ss << "Current device: " << currentDevice << " " << cv::cuda::DeviceInfo(currentDevice).name();
        ss << " - Num available devices: " << maxDevice;
        log(Status, ss.str());
        updateParameter<std::string>("Device name", cv::cuda::DeviceInfo(currentDevice).name());
        firstRun = false;
    }

    unsigned int device = getParameter<unsigned int>(0)->data;
    if(device >= maxDevice)
    {
        log(Status, "Desired device greater than max allowed device index, max index: " + boost::lexical_cast<std::string>(maxDevice - 1));
        return cv::cuda::GpuMat();
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);
    if(currentDevice != device)
    {
        log(Status, "Switching device from " + boost::lexical_cast<std::string>(currentDevice) + " to " + boost::lexical_cast<std::string>(device) + " " + prop.name);
        updateParameter<std::string>("Device name", cv::cuda::DeviceInfo(device).name());
    }
    cv::cuda::setDevice(device);
    return cv::cuda::GpuMat();
}
bool SetDevice::SkipEmpty() const
{
    return false;

}

NODE_DEFAULT_CONSTRUCTOR_IMPL(SetDevice)
