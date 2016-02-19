#include "ParameteredObject.h"
#include <EagleLib/rcc/SystemTable.hpp>
#include "ObjectInterfacePerModule.h"
#include "remotery/lib/Remotery.h"
#include "Signals.h"
#include "ParameteredObjectImpl.hpp"
#include <EagleLib/Signals.h>
#include <Signals/logging.hpp>

using namespace EagleLib;
using namespace Parameters;


ParameteredObject::ParameteredObject()
{
	_sig_parameter_updated = nullptr;
	_sig_parameter_added = nullptr;
}

ParameteredObject::~ParameteredObject()
{
    
}

void ParameteredObject::setup_signals(EagleLib::SignalManager* manager)
{
	_sig_parameter_updated = manager->get_signal<void(ParameteredObject*)>("parameter_updated", this, "Emitted when a parameter is updated from ui");
	_sig_parameter_added = manager->get_signal<void(ParameteredObject*)>("parameter_added", this, "Emitted when a new parameter is added");
}

void ParameteredIObject::Serialize(ISimpleSerializer* pSerializer)
{
    IObject::Serialize(pSerializer);
    SERIALIZE(parameters);
	SERIALIZE(_sig_parameter_updated);
	SERIALIZE(_sig_parameter_added);
}
void ParameteredIObject::Init(const cv::FileNode& configNode)
{
    
}
void ParameteredIObject::Init(bool firstInit)
{
	if (firstInit)
	{

	}
	else
	{
		for (auto& param : parameters)
		{
			_callback_connections.push_back(param->RegisterNotifier(std::bind(&ParameteredIObject::onUpdate, this, std::placeholders::_1)));
		}
	}
}

Parameter* ParameteredObject::addParameter(Parameter::Ptr param)
{
    parameters.push_back(param);
	std::lock_guard<std::recursive_mutex> lock(mtx);
	DOIF_LOG_FAIL(_sig_parameter_added, (*_sig_parameter_updated)(this), warning);
	_callback_connections.push_back(param->RegisterNotifier(std::bind(&ParameteredObject::onUpdate, this, std::placeholders::_1)));
    return param.get();
}

Parameter::Ptr ParameteredObject::getParameter(int idx)
{
    CV_Assert(idx >= 0 && idx < parameters.size());
    return parameters[idx];
}

Parameter::Ptr ParameteredObject::getParameter(const std::string& name)
{
    for (auto& itr : parameters)
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
    if (idx < 0 || idx >= parameters.size())
    {
        BOOST_LOG_TRIVIAL(debug) << "Requested index " << idx << " out of bounds " << parameters.size();
        return Parameter::Ptr();
    }
    return parameters[idx];
}

Parameter::Ptr ParameteredObject::getParameterOptional(const std::string& name)
{
    for (auto& itr : parameters)
    {
        if (itr->GetName() == name)
        {
            return itr;
        }
    }
    BOOST_LOG_TRIVIAL(debug) << "Unable to find parameter by name: " << name;
    return Parameter::Ptr();
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
void ParameteredObject::onUpdate(cv::cuda::Stream* stream)
{
	DOIF(_sig_parameter_updated != nullptr, (*_sig_parameter_updated)(this), debug);
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
    return index < parameters.size();
}
