#pragma once

namespace EagleLib
{
    struct Algorithm::impl
    {
        long long ts = -1;    
        long long last_ts = -1;
        mo::InputParameter* sync_input = nullptr;
    };
}