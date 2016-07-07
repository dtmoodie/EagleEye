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
    SERIALIZE(_implicit_parameters);
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
    _callback_connections.clear();
    IObject::Init(firstInit);
    
    if (firstInit)
    {
        InitializeExplicitParamsToDefault();
        WrapExplicitParams();
    }
    else
    {
        WrapExplicitParams();
        _parameters.clear();
        for(auto& param : _implicit_parameters)
        {
            _parameters.push_back(param.get());
        }
        for(auto& param : _explicit_parameters)
        {
            _parameters.push_back(param);
        }
        for (auto& param : _parameters)
        {
            RegisterParameterCallback(param, std::bind(&ParameteredIObject::onUpdate, this, param, std::placeholders::_1));
            DOIF_LOG_FAIL(_variable_manager, _variable_manager->AddParameter(param), debug);
        }
        sig_object_recompiled(this);
    }
    
}
