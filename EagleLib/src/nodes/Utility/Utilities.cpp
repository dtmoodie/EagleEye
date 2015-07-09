#include "nodes/Utility/Utilities.h"




using namespace EagleLib;

void SyncFunctionCall::Init(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Call all input functions", boost::bind(&SyncFunctionCall::call, this));
    if(firstInit)
    {
        addInputParameter<boost::function<void(void)>>("Input 1");
    }
}

void SyncFunctionCall::call()
{
    for(int i = 1; i < parameters.size(); ++i)
    {
		auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<boost::function<void(void)>>>(parameters[i]);
        if(param)
        {
            if(param->Data() != nullptr)
                (*param->Data())();
        }
    }
}

cv::cuda::GpuMat SyncFunctionCall::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    bool full = true;
    for(int i = 1; i < parameters.size(); ++i)
    {
        auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<boost::function<void(void)>>>(parameters[i]);
		
        if(param)
        {
            if(param->Data() == nullptr)
                full = false;
        }
    }
    if(full == true)
    {
        addInputParameter<boost::function<void(void)>>("Input " + boost::lexical_cast<std::string>(parameters.size()));
    }
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(SyncFunctionCall)
