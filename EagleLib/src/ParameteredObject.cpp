#include "ParameteredObject.h"
#include <SystemTable.hpp>
#include "ObjectInterfacePerModule.h"
#include "Events.h"
using namespace EagleLib;
using namespace Parameters;
namespace EagleLib
{
    struct ParameteredObjectImpl
    {
        ParameteredObjectImpl()
        {

        }
        ~ParameteredObjectImpl()
        {
            for (auto itr : callbackConnections)
            {
                for (auto itr2 : itr.second)
                {
                    itr2.disconnect();
                }
            }
        }
        std::map<ParameteredObject*, std::list<boost::signals2::connection>>	callbackConnections;
        boost::recursive_mutex mtx;
    };

}


ParameteredObject::ParameteredObject():
    _impl(new ParameteredObjectImpl())
{

}

ParameteredObject::~ParameteredObject()
{

}

void ParameteredObject::Serialize(ISimpleSerializer* pSerializer)
{
    IObject::Serialize(pSerializer);
    SERIALIZE(_impl);
    SERIALIZE(parameters);
}
void ParameteredObject::Init(const cv::FileNode& configNode)
{
    
}

Parameter* ParameteredObject::addParameter(Parameter::Ptr param)
{
    parameters.push_back(param);
    boost::recursive_mutex::scoped_lock lock(_impl->mtx);
    _impl->callbackConnections[this].push_back(param->RegisterNotifier(boost::bind(&ParameteredObject::onUpdate, this, _1)));
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
void ParameteredObject::RegisterParameterCallback(int idx, const boost::function<void(cv::cuda::Stream*)>& callback, bool lock_param, bool lock_object)
{
    RegisterParameterCallback(getParameter(idx).get(), callback, lock_param, lock_object);
}

void ParameteredObject::RegisterParameterCallback(const std::string& name, const boost::function<void(cv::cuda::Stream*)>& callback, bool lock_param, bool lock_object)
{
    RegisterParameterCallback(getParameter(name).get(), callback, lock_param, lock_object);
}

void ParameteredObject::RegisterParameterCallback(Parameter* param, const boost::function<void(cv::cuda::Stream*)>& callback, bool lock_param, bool lock_object)
{
    if (lock_param && !lock_object)
    {
        _impl->callbackConnections[this].push_back(param->RegisterNotifier(boost::bind(&ParameteredObject::RunCallbackLockParameter, this, _1, callback, &param->mtx)));
        return;
    }
    if (lock_object && !lock_param)
    {
        _impl->callbackConnections[this].push_back(param->RegisterNotifier(boost::bind(&ParameteredObject::RunCallbackLockObject, this, _1, callback)));
        return;
    }
    if (lock_object && lock_param)
    {

        _impl->callbackConnections[this].push_back(param->RegisterNotifier(boost::bind(&ParameteredObject::RunCallbackLockBoth, this, _1, callback, &param->mtx)));
        return;
    }
    _impl->callbackConnections[this].push_back(param->RegisterNotifier(callback));
    
}
void ParameteredObject::onUpdate(cv::cuda::Stream* stream)
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto signalHandler = table->GetSingleton<ISignalHandler>();
        auto signal = signalHandler->GetSignalSafe<boost::signals2::signal<void(IObject*)>>("ObjectUpdated");
        (*signal)(this);
    }
}
void ParameteredObject::RunCallbackLockObject(cv::cuda::Stream* stream, const boost::function<void(cv::cuda::Stream*)>& callback)
{
    boost::recursive_mutex::scoped_lock lock(mtx);
    callback(stream);
}
void ParameteredObject::RunCallbackLockParameter(cv::cuda::Stream* stream, const boost::function<void(cv::cuda::Stream*)>& callback, boost::recursive_mutex* paramMtx)
{
    boost::recursive_mutex::scoped_lock lock(*paramMtx);
    callback(stream);
}
void ParameteredObject::RunCallbackLockBoth(cv::cuda::Stream* stream, const boost::function<void(cv::cuda::Stream*)>& callback, boost::recursive_mutex* paramMtx)
{
    boost::recursive_mutex::scoped_lock lock(mtx);
    boost::recursive_mutex::scoped_lock param_lock(*paramMtx);
    callback(stream);
}