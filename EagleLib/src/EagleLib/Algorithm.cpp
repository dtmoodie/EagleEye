#include "EagleLib/Algorithm.h"
#include <MetaObject/Parameters/InputParameter.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>

using namespace mo;
using namespace EagleLib;


struct Algorithm::impl
{
    long long ts = -1;    
    mo::InputParameter* sync_input = nullptr;
};

Algorithm::Algorithm()
{
    _pimpl = new impl();
    _enabled = true;
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

void Algorithm::Process()
{
    if(_enabled == false)
        return;
    
    ProcessImpl();

    if(_pimpl->sync_input == nullptr && _pimpl->ts != -1)
        ++_pimpl->ts;
}

bool Algorithm::CheckInputs()
{
    auto inputs = this->GetInputs();
    if(inputs.size() == 0)
        return true;
    if(_pimpl->sync_input != nullptr)
    {
        _pimpl->ts = dynamic_cast<IParameter*>(_pimpl->sync_input)->GetTimestamp();
    }
    if(_pimpl->ts == -1)
    {
        for(auto input : inputs)
        {
            long long ts = dynamic_cast<IParameter*>(input)->GetTimestamp();
            if(ts != -1)
            {
                if(_pimpl->ts == -1)
                    _pimpl->ts = ts;
                else
                    _pimpl->ts = std::min(_pimpl->ts, ts);
            }
        }
    }
    for(auto input : inputs)
    {
        if(!input->GetInput(_pimpl->ts))
        {
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