#include "nodes/Utility/Cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


RUNTIME_COMPILER_LINKLIBRARY("-lcudart")
using namespace EagleLib;

void SetDevice::Init(bool firstInit)
{
	if (firstInit)
	{
        updateParameter<unsigned int>("Device Number", uint32_t(0));
		updateParameter<std::string>("Device name", "")->type = Parameters::Parameter::State;
		firstRun = true;
        currentDevice = cv::cuda::getDevice();
        maxDevice = cv::cuda::getCudaEnabledDeviceCount();
        auto deviceInfo = cv::cuda::DeviceInfo(currentDevice);
        updateParameter<std::string>("Device name", deviceInfo.name());
	}    
}

cv::cuda::GpuMat SetDevice::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{

    if(parameters[0]->changed)
    {
        unsigned int device = *getParameter<unsigned int>(0)->Data();
        if(device >= maxDevice)
        {
            //log(Status, "Desired device greater than max allowed device index, max index: " + boost::lexical_cast<std::string>(maxDevice - 1));
            NODE_LOG(info) << "Desired device greater than max allowed device index, max index: "  << maxDevice - 1;
            return cv::cuda::GpuMat();
        }
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,device);
        if(currentDevice != device)
        {
            //log(Status, "Switching device from " + boost::lexical_cast<std::string>(currentDevice) + " to " + boost::lexical_cast<std::string>(device) + " " + prop.name + " async engines: " + boost::lexical_cast<std::string>(prop.asyncEngineCount));
            NODE_LOG(info) << "Switching device from " << currentDevice << " to " << device << " " << prop.name << " async engines: " << prop.asyncEngineCount;
            updateParameter<std::string>("Device name", cv::cuda::DeviceInfo(device).name());
            //if(onUpdate)
            //    onUpdate(this);
            cv::cuda::setDevice(device);
            stream = cv::cuda::Stream();
            return cv::cuda::GpuMat();
        }
        parameters[0]->changed = false;
    }


	return img;
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
        streams.resize(*getParameter<int>(0)->Data());
    }
    stream = *streams.getFront();
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(SetDevice, Utility)

