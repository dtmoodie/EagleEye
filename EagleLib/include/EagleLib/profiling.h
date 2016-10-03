#pragma once
#include "EagleLib/Detail/Export.hpp"

namespace EagleLib
{
    EAGLE_EXPORTS void InitProfiling();
    struct EAGLE_EXPORTS scoped_profile
    {
        scoped_profile(const char* name);
        scoped_profile(const char* name, const char* func);
        ~scoped_profile();
    };
}


#define PROFILE_OBJ(name) EagleLib::scoped_profile    profile_object(name, __FUNCTION__)
#define PROFILE_RANGE(name) EagleLib::scoped_profile    profile_scope_##name(#name)
#define PROFILE_FUNCTION EagleLib::scoped_profile        profile_function(__FUNCTION__);
