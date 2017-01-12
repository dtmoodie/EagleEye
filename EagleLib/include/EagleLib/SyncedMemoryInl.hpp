#pragma once

#include "SyncedMemory.h"
namespace EagleLib
{
    template<typename A> void SyncedMemory::load(A& ar)
    {
        ar(cereal::make_nvp("matrices", h_data));
        sync_flags.resize(h_data.size(), HOST_UPDATED);
        d_data.resize(h_data.size());
    }
    template<typename A> void SyncedMemory::save(A & ar) const
    {
        ar(cereal::make_nvp("matrices", h_data));
    }
}
