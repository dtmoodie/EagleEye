#pragma once
#include "EagleLib/Defs.hpp"
struct EAGLE_EXPORTS scoped_profile
{
	scoped_profile(const char* name);
	scoped_profile(const char* name, const char* func);
	~scoped_profile();
};

#define PROFILE_OBJ(name) scoped_profile	profile_object(name, __FUNCTION__)
#define PROFILE_RANGE(name) scoped_profile	profile_scope_##name(#name)
#define PROFILE_FUNCTION scoped_profile		profile_function(__FUNCTION__);