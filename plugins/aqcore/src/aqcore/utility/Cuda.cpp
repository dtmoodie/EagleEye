#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "Cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

/*RUNTIME_COMPILER_LINKLIBRARY("-lcudart")
using namespace aq;
using namespace aq::nodes;

void SetDevice::nodeInit(bool firstInit)
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

    if(_parameters[0]->changed)
    {
        unsigned int device = *getParameter<unsigned int>(0)->Data();
        if(device >= maxDevice)
        {
            //log(Status, "Desired device greater than max allowed device index, max index: " +
boost::lexical_cast<std::string>(maxDevice - 1));
            MO_LOG(info) << "Desired device greater than max allowed device index, max index: "  << maxDevice - 1;
            return cv::cuda::GpuMat();
        }
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,device);
        if(currentDevice != device)
        {
            //log(Status, "Switching device from " + boost::lexical_cast<std::string>(currentDevice) + " to " +
boost::lexical_cast<std::string>(device) + " " + prop.name + " async engines: " +
boost::lexical_cast<std::string>(prop.asyncEngineCount));
            MO_LOG(info) << "Switching device from " << currentDevice << " to " << device << " " << prop.name << " async
engines: " << prop.asyncEngineCount;
            updateParameter<std::string>("Device name", cv::cuda::DeviceInfo(device).name());
            //if(onUpdate)
            //    onUpdate(this);
            cv::cuda::setDevice(device);
            stream = cv::cuda::stream();
            return cv::cuda::GpuMat();
        }
        _parameters[0]->changed = false;
    }


    return img;
}
bool SetDevice::SkipEmpty() const
{
    return false;
}



MO_REGISTER_CLASS(SetDevice, Utility)*/

#endif