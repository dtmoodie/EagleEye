#include "EagleLib/Algorithm.h"
#include "EagleLib/Detail/AlgorithmImpl.hpp"
#include <MetaObject/Parameters/InputParameter.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>

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

bool Algorithm::Process()
{
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    if(_enabled == false)
        return false;
    if(!CheckInputs())
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

bool Algorithm::CheckInputs()
{
    auto inputs = this->GetInputs();
    if(inputs.size() == 0)
        return true;
    long long ts = -1;
    if(_pimpl->ts == -1 && _pimpl->sync_input == nullptr)
    {
        for(auto input : inputs)
        {
            long long ts = input->GetTimestamp();
            if(ts != -1)
            {
                if(_pimpl->ts == -1)
                    _pimpl->ts = ts;
                else
                    _pimpl->ts = std::min(_pimpl->ts, ts);
            }
        }
        if(_pimpl->ts != -1)
            LOG(trace) << "Timestamp updated to " << _pimpl->ts;
        ts = _pimpl->ts;
    }
    if(_pimpl->_sync_method == SyncEvery && _pimpl->sync_input)
    {
        if(_pimpl->_ts_processing_queue.size() != 0)
        {
            ts = _pimpl->_ts_processing_queue.front();
            _pimpl->ts = ts;
        }else
        {
            LOG(trace) << "No new data to be processed";
            return false; // no new data to be processed
        }
    }else if(_pimpl->_sync_method == SyncNewest && _pimpl->sync_input)
    {
        ts = _pimpl->ts;
        if(ts == _pimpl->last_ts)
            return false;
    }
    
    for(auto input : inputs)
    {
        if(!input->GetInput(ts) && !input->CheckFlags(Optional_e))
        {
            LOG(trace) << input->GetTreeName() << " failed to get input at timestamp: " << ts;
            return false;
        }
    }
    return true;
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
}

void Algorithm::SetSyncMethod(SyncMethod _method)
{
    if(_pimpl->_sync_method == SyncEvery && _method != SyncEvery)
    {
        std::swap(_pimpl->_ts_processing_queue, std::queue<long long>());
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