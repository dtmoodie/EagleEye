#include "nodes/Utility/Cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


RUNTIME_COMPILER_LINKLIBRARY("-lcudart")
using namespace EagleLib;

void SetDevice::Init(bool firstInit)
{
    updateParameter<unsigned int>("Device Number", 0);
    updateParameter<std::string>("Device name","", Parameter::State);
    firstRun = true;
}

cv::cuda::GpuMat SetDevice::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    cv::Mat asd;
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
        if(onUpdate)
            onUpdate(this);
    }
    cv::cuda::setDevice(device);
    return cv::cuda::GpuMat();
}
bool SetDevice::SkipEmpty() const
{
    return false;
}


bool StreamDispatcher::SkipEmpty() const
{
    return false;
}
void StreamDispatcher::Init(bool firstInit)
{
    if(firstInit)
    {
        updateParameter("Number of streams", int(20));
    }
}

cv::cuda::GpuMat StreamDispatcher::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(parameters[0]->changed)
    {
        streams.resize(getParameter<int>(0)->data);
    }
    stream = *streams.getFront();
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(SetDevice)
