#pragma once

#include "SyncedMemory.h"
namespace EagleLib
{
    template<typename A> void SyncedMemory::load(A& ar)
    {
        ar(cereal::make_nvp("matrices", _pimpl->h_data));
        _pimpl->sync_flags.resize(_pimpl->h_data.size(), HOST_UPDATED);
        _pimpl->d_data.resize(_pimpl->h_data.size());
    }
    template<typename A> void SyncedMemory::save(A & ar) const
    {
        this->Synchronize();
        ar(cereal::make_nvp("matrices", _pimpl->h_data));
    }
}
