#pragma once
#include "EagleLib/Algorithm.h"
#include <queue>
namespace EagleLib
{
    struct Algorithm::impl
    {
        long long ts = -1;    
        long long last_ts = -1;
        mo::InputParameter* sync_input = nullptr;
        Algorithm::SyncMethod _sync_method;
        std::queue<long long> _ts_processing_queue;
    };
}