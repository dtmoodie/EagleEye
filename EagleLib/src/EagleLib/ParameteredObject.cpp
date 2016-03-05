#include "ParameteredObject.h"
#include <EagleLib/rcc/SystemTable.hpp>
#include "ObjectInterfacePerModule.h"
#include "remotery/lib/Remotery.h"
#include "Signals.h"
#include "ParameteredObjectImpl.hpp"
#include <EagleLib/Signals.h>
#include <signals/logging.hpp>
#include <EagleLib/VariableManager.h>

using namespace EagleLib;
using namespace Parameters;


ParameteredObject::ParameteredObject()
{
	_sig_parameter_updated = nullptr;
	_sig_parameter_added = nullptr;
    _variable_manager = nullptr;
}

ParameteredObject::~ParameteredObject()
{
    _callback_connections.clear();
    if(_variable_manager)
    {
        for(int i = 0; i < _parameters.size(); ++i)
        {
            _variable_manager->RemoveParameter(_parameters[i]);
        }
    }
    _parameters.clear();
}

void ParameteredObject::setup_signals(SignalManager* manager)
{
	_sig_parameter_updated = manager->get_signal<void(ParameteredObject*)>("parameter_updated", this, "Emitted when a parameter is updated from ui");
	_sig_parameter_added = manager->get_signal<void(ParameteredObject*)>("parameter_added", this, "Emitted when a new parameter is added");
}
void ParameteredObject::SetupVariableManager(IVariableManager* manager)
{
    _variable_manager = manager;
}
IVariableManager* ParameteredObject::GetVariableManager()
{
    return _variable_manager;
}
ParameteredIObject::ParameteredIObject()
{
    
}

void ParameteredIObject::Serialize(ISimpleSerializer* pSerializer)
{
    IObject::Serialize(pSerializer);
    SERIALIZE(_parameters);
	SERIALIZE(_sig_parameter_updated);
	SERIALIZE(_sig_parameter_added);
    SERIALIZE(_variable_manager);
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
            RegisterParameterCallback(param.get(), std::bind(&ParameteredIObject::onUpdate, this, param.get(), std::placeholders::_1));
            DOIF_LOG_FAIL(_variable_manager, _variable_manager->AddParameter(param), debug);
		}		
	}
}

Parameter* ParameteredObject::addParameter(Parameter::Ptr param)
{
	std::lock_guard<std::recursive_mutex> lock(mtx);
	DOIF_LOG_FAIL(_sig_parameter_added, (*_sig_parameter_updated)(this), warning);
    DOIF_LOG_FAIL(_variable_manager, _variable_manager->AddParameter(param), debug);
	_callback_connections.push_back(param->RegisterNotifier(std::bind(&ParameteredObject::onUpdate, this, param.get(), std::placeholders::_1)));
    _parameters.push_back(param);
    return param.get();
}
void ParameteredObject::RemoveParameter(std::string name)
{
    for(auto itr = _parameters.begin(); itr != _parameters.end(); ++itr)
    {
        if((*itr)->GetName() == name)
        {
            DOIF_LOG_FAIL(_variable_manager, _variable_manager->RemoveParameter(*itr), debug);
            itr = _parameters.erase(itr);
        }
    }
}
void ParameteredObject::RemoveParameter(size_t index)
{
    if(index < _parameters.size())
    {
        DOIF_LOG_FAIL(_variable_manager, _variable_manager->RemoveParameter(_parameters[index]), debug);
        _parameters.erase(_parameters.begin() + index);
    }
}

Parameter::Ptr ParameteredObject::getParameter(int idx)
{
    CV_Assert(idx >= 0 && idx < _parameters.size());
    return _parameters[idx];
}

Parameter::Ptr ParameteredObject::getParameter(const std::string& name)
{
    for (auto& itr : _parameters)
    {
        if (itr->GetName() == name)
        {
            return itr;
        }
    }
    throw std::string("Unable to find parameter by name: " + name);
}

Parameter::Ptr ParameteredObject::getParameterOptional(int idx)
{
    if (idx < 0 || idx >= _parameters.size())
    {
        BOOST_LOG_TRIVIAL(debug) << "Requested index " << idx << " out of bounds " << _parameters.size();
        return Parameter::Ptr();
    }
    return _parameters[idx];
}

Parameter::Ptr ParameteredObject::getParameterOptional(const std::string& name)
{
    for (auto& itr : _parameters)
    {
        if (itr->GetName() == name)
        {
            return itr;
        }
    }
    BOOST_LOG_TRIVIAL(debug) << "Unable to find parameter by name: " << name;
    return Parameter::Ptr();
}
std::vector<ParameterPtr> ParameteredObject::getParameters()
{
    return _parameters;
}


void ParameteredObject::RegisterParameterCallback(int idx, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param, bool lock_object)
{
    RegisterParameterCallback(getParameter(idx).get(), callback, lock_param, lock_object);
}

void ParameteredObject::RegisterParameterCallback(const std::string& name, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param, bool lock_object)
{
    RegisterParameterCallback(getParameter(name).get(), callback, lock_param, lock_object);
}

void ParameteredObject::RegisterParameterCallback(Parameter* param, const std::function<void(cv::cuda::Stream*)>& callback, bool lock_param, bool lock_object)
{
    if (lock_param && !lock_object)
    {
        _callback_connections.push_back(param->RegisterNotifier(std::bind(&ParameteredObject::RunCallbackLockParameter, this, std::placeholders::_1, callback, &param->mtx)));
        return;
    }
    if (lock_object && !lock_param)
    {
		_callback_connections.push_back(param->RegisterNotifier(std::bind(&ParameteredObject::RunCallbackLockObject, this, std::placeholders::_1, callback)));
        return;
    }
    if (lock_object && lock_param)
    {

		_callback_connections.push_back(param->RegisterNotifier(std::bind(&ParameteredObject::RunCallbackLockBoth, this, std::placeholders::_1, callback, &param->mtx)));
        return;
    }
	_callback_connections.push_back(param->RegisterNotifier(callback));
    
}
void ParameteredObject::onUpdate(Parameters::Parameter* param, cv::cuda::Stream* stream)
{
	DOIF_LOG_FAIL(_sig_parameter_updated != nullptr, (*_sig_parameter_updated)(this), debug);
}
void ParameteredObject::RunCallbackLockObject(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback)
{
	rmt_ScopedCPUSample(ParameteredObject_RunCallbackLockObject);
	std::lock_guard<std::recursive_mutex> lock(mtx);
    callback(stream);
}
void ParameteredObject::RunCallbackLockParameter(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback, std::recursive_mutex* paramMtx)
{
	rmt_ScopedCPUSample(ParameteredObject_RunCallbackLockParameter);
	std::lock_guard<std::recursive_mutex> lock(*paramMtx);
    callback(stream);
}
void ParameteredObject::RunCallbackLockBoth(cv::cuda::Stream* stream, const std::function<void(cv::cuda::Stream*)>& callback, std::recursive_mutex* paramMtx)
{
	rmt_ScopedCPUSample(ParameteredObject_RunCallbackLockBoth);
	std::lock_guard<std::recursive_mutex> lock(mtx);
	std::lock_guard<std::recursive_mutex> lock_(*paramMtx);
    callback(stream);
}
bool ParameteredObject::exists(const std::string& name)
{
    return getParameterOptional(name) != nullptr;
}
bool ParameteredObject::exists(size_t index)
{
    return index < _parameters.size();
}
