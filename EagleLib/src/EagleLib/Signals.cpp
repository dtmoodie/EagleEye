#include "Signals.h"


using namespace EagleLib;

std::shared_ptr<Signals::signal_base>& SignalManager::GetSignal(const std::string& name, Loki::TypeInfo type, int stream_index)
{
    if(stream_index == -1)
    {
        //return _signals[type][name];
        return get_signal(name, type);
    }
    return stream_specific_signals[stream_index][type][name];
}