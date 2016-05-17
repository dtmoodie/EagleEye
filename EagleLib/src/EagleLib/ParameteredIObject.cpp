#include "ParameteredIObject.h"
#include "signals/logging.hpp"
#include "parameters/IVariableManager.h"
using namespace EagleLib;
ParameteredIObject::ParameteredIObject()
{

}

void ParameteredIObject::Serialize(ISimpleSerializer* pSerializer)
{
	IObject::Serialize(pSerializer);
	SerializeAllParams(pSerializer);
	SERIALIZE(_parameters);
	//SERIALIZE(_sig_parameter_updated);
	//SERIALIZE(_sig_parameter_added);
	SERIALIZE(_variable_manager);
}

void ParameteredIObject::SerializeAllParams(ISimpleSerializer* pSerializer)
{

}

void ParameteredIObject::Init(const cv::FileNode& configNode)
{

}

void ParameteredIObject::Init(bool firstInit)
{
	IObject::Init(firstInit);
	if (firstInit)
	{

	}
	else
	{
		for (auto& param : _parameters)
		{
			RegisterParameterCallback(param, std::bind(&ParameteredIObject::onUpdate, this, param, std::placeholders::_1));
			DOIF_LOG_FAIL(_variable_manager, _variable_manager->AddParameter(param), debug);
		}
	}
}
