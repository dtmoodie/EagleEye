#include "nodes/Utility/Utilities.h"
#include <boost/lexical_cast.hpp>
#include <EagleLib/ParameteredObjectImpl.hpp>



using namespace EagleLib;
using namespace EagleLib::Nodes;

bool functionQualifier(Parameters::Parameter* parameter)
{
	if (parameter->GetTypeInfo() == Loki::TypeInfo(typeid(boost::function<void(void)>)))
	{
		if (parameter->type & Parameters::Parameter::Output || parameter->type & Parameters::Parameter::Control)
			return true;

	}
	return false;
}

void SyncFunctionCall::Init(bool firstInit)
{
    updateParameter<boost::function<void(void)>>("Call all input functions", boost::bind(&SyncFunctionCall::call, this));
    if(firstInit)
    {
		addInputParameter<boost::function<void(void)>>("Input 1")->SetQualifier(boost::bind(&functionQualifier, _1));
    }
}

void SyncFunctionCall::call()
{
    for(int i = 1; i < _parameters.size(); ++i)
    {
		auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<boost::function<void(void)>>>(_parameters[i]);
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
    for(int i = 1; i < _parameters.size(); ++i)
    {
        auto param = std::dynamic_pointer_cast<Parameters::ITypedParameter<boost::function<void(void)>>>(_parameters[i]);
		
        if(param)
        {
            if(param->Data() == nullptr)
                full = false;
        }
    }
    if(full == true)
    {
		addInputParameter<boost::function<void(void)>>("Input " + boost::lexical_cast<std::string>(_parameters.size()))->SetQualifier(boost::bind(&functionQualifier, _1));
    }
    return img;
}


NODE_DEFAULT_CONSTRUCTOR_IMPL(SyncFunctionCall, Utility)
