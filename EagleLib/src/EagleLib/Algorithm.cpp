#include <MetaObject/Parameters/IO/CerealPolicy.hpp>
#include <MetaObject/Parameters/IO/CerealMemory.hpp>

#include "EagleLib/Algorithm.h"
#include "EagleLib/Detail/AlgorithmImpl.hpp"
#include <MetaObject/Parameters/InputParameter.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>

#include <EagleLib/IO/JsonArchive.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

INSTANTIATE_META_PARAMETER(std::vector<rcc::shared_ptr<EagleLib::Algorithm>>)
using namespace mo;
using namespace EagleLib;


Algorithm::Algorithm()
{
    _pimpl = new impl();
    _enabled = true;
    _pimpl->_sync_method = SyncEvery;
}

Algorithm::~Algorithm()
{
    delete _pimpl;
    _pimpl = nullptr;
}

double Algorithm::GetAverageProcessingTime() const
{
    return 0.0;
}

void Algorithm::SetEnabled(bool value)
{
    _enabled = value;
}

bool Algorithm::IsEnabled() const
{
    return _enabled;
}
std::vector<mo::IParameter*> Algorithm::GetComponentParameters(const std::string& filter) const
{
    std::vector<mo::IParameter*> output; // = mo::IMetaObject::GetParameters(filter);
    for(auto& component: _algorithm_components)
    {
        if(component)
        {
            std::vector<mo::IParameter*> output2 = component->GetParameters(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}
std::vector<mo::IParameter*> Algorithm::GetAllParameters(const std::string& filter) const
{
    std::vector<mo::IParameter*> output = mo::IMetaObject::GetParameters(filter);
    for(auto& component: _algorithm_components)
    {
        if(component)
        {
            std::vector<mo::IParameter*> output2 = component->GetParameters(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}
bool Algorithm::Process()
{
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    if(_enabled == false)
        return false;
    if(CheckInputs() == NoneValid)
    {
        return false;
    }
    if(ProcessImpl())
    {
        _pimpl->last_ts = _pimpl->ts;
        if(_pimpl->sync_input == nullptr && _pimpl->ts != -1)
            ++_pimpl->ts;
        if(_pimpl->_sync_method == SyncEvery && _pimpl->sync_input)
        {
            if(_pimpl->ts == _pimpl->_ts_processing_queue.front())
            {
                _pimpl->_ts_processing_queue.pop();
            }
        }
        return true;
    }
    return false;
}
mo::IParameter* Algorithm::GetOutput(const std::string& name) const
{
    auto output = mo::IMetaObject::GetOutput(name);
    if(output)
        return output;
    if(!output)
    {
        for(auto& component: _algorithm_components)
        {
            if(component)
            {
                output = component->GetOutput(name);
                if(output)
                    return output;
            }
        }
    }
    return nullptr;
}



Algorithm::InputState Algorithm::CheckInputs()
{
    auto inputs = this->GetInputs();
    if(inputs.size() == 0)
        return AllValid;

    long long ts = -1;
    if(_pimpl->sync_input == nullptr)
    {
        for(auto input : inputs)
        {
            long long ts_in = input->GetTimestamp();
            if(ts_in != -1)
            {
                if(ts == -1)
                    ts = ts_in;
                else
                    ts = std::min(ts, ts_in);
            }
        }
        if(ts != -1)
            LOG(trace) << "Timestamp updated to " << _pimpl->ts;
    }
    //ts = _pimpl->ts;
    if(_pimpl->_sync_method == SyncEvery && _pimpl->sync_input)
    {
        if(_pimpl->_ts_processing_queue.size() != 0)
        {
            ts = _pimpl->_ts_processing_queue.front();
            _pimpl->ts = ts;
        }else
        {
            LOG(trace) << "No new data to be processed";
            // TODO TEST
            return NotUpdated;
        }
    }else if(_pimpl->_sync_method == SyncNewest && _pimpl->sync_input)
    {
        ts = _pimpl->ts;
        if(ts == _pimpl->last_ts)
            return NoneValid;
    }

    for(auto input : inputs)
    {
        if(!input->GetInput(ts))
        {
            if(input->CheckFlags(Optional_e))
            {
                // If the input isn't set and it's optional then this is ok
                if(input->GetInputParam())
                {
                    // Input is optional and set, but couldn't get the right timestamp, error
                    LOG(debug) << "Failed to get input \"" << input->GetTreeName() << "\" at timestamp " << ts;
                    //return false;
                }else
                {
                    LOG(trace) << "Optional input not set \"" << input->GetTreeName() << "\"";
                }
            }else
            {
                // Input is not optional
                if (input->GetInputParam())
                {
                    LOG(debug) << "Failed to get input \"" << input->GetTreeName() << "\" at timestamp " << ts;
                    return NoneValid;
                }else
                {
                    LOG(debug) << "Input not set \"" << input->GetTreeName() << "\"";
                    return NoneValid;
                }
            }
        }
    }
    _pimpl->ts = ts;
    if(ts == _pimpl->last_ts)
        return NotUpdated;
    return AllValid;
}

void Algorithm::Clock(int line_number)
{
    
}

long long Algorithm::GetTimestamp() 
{
    return _pimpl->ts;
}

void Algorithm::SetSyncInput(const std::string& name)
{
    _pimpl->sync_input = GetInput(name);
    if(_pimpl->sync_input)
    {
        LOG(info) << "Updating sync parameter for " << this->GetTypeName() << " to " << name;
    }else
    {
        LOG(warning) << "Unable to set sync input for " << this->GetTypeName() << " to " << name;
    }
}

void Algorithm::SetSyncMethod(SyncMethod _method)
{
    if(_pimpl->_sync_method == SyncEvery && _method != SyncEvery)
    {
        //std::swap(_pimpl->_ts_processing_queue, std::queue<long long>());
        _pimpl->_ts_processing_queue = std::queue<long long>();
    }
    _pimpl->_sync_method = _method;
    
}
void Algorithm::onParameterUpdate(mo::Context* ctx, mo::IParameter* param)
{
    mo::IMetaObject::onParameterUpdate(ctx, param);
    if(_pimpl->_sync_method == SyncEvery)
    {
        if(param == _pimpl->sync_input)
        {
            long long ts = param->GetTimestamp();
            boost::recursive_mutex::scoped_lock lock(_pimpl->_mtx);
#ifdef _MSC_VER
#ifdef _DEBUG
            _pimpl->timestamps.push_back(ts);
            if(_pimpl->_ts_processing_queue.size() && ts != (_pimpl->_ts_processing_queue.back() + 1))
                LOG(debug) << "Timestamp not monotonically incrementing.  Current: " << ts << " previous: " << _pimpl->_ts_processing_queue.back();
            auto itr = std::find(_pimpl->_ts_processing_queue._Get_container().begin(), _pimpl->_ts_processing_queue._Get_container().end(), ts);
            if(itr != _pimpl->_ts_processing_queue._Get_container().end())
            {
                LOG(debug) << "Timestamp (" << ts << ") exists in processing queue.";
            }
#endif
#endif
            _pimpl->_ts_processing_queue.push(ts);
        }
    }else if (_pimpl->_sync_method == SyncNewest)
    {
        if(param == _pimpl->sync_input)
        {
            _pimpl->ts = param->GetTimestamp();
        }
    }
}
void  Algorithm::SetContext(mo::Context* ctx, bool overwrite)
{
    mo::IMetaObject::SetContext(ctx, overwrite);
    for(auto& child : _algorithm_components)
    {
        child->SetContext(ctx, overwrite);
    }
}

void Algorithm::PostSerializeInit()
{
    for(auto& child : _algorithm_components)
    {
        child->SetContext(this->_ctx);
        child->PostSerializeInit();
    }
}
void Algorithm::AddComponent(rcc::weak_ptr<Algorithm> component)
{
    _algorithm_components.push_back(component);
    mo::ISlot* slot = this->GetSlot("parameter_updated", mo::TypeInfo(typeid(void(mo::Context*, mo::IParameter*))));
    if(slot)
    {
        auto params = component->GetParameters();
        for(auto param : params)
        {
            param->RegisterUpdateNotifier(slot);
        }
    }


}
void  Algorithm::Serialize(ISimpleSerializer *pSerializer)
{
    mo::IMetaObject::Serialize(pSerializer);
    SERIALIZE(_algorithm_components);
}
