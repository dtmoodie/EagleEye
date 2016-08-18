#include "EagleLib/profiling.h"
#include <string>
#include <sstream>
#ifdef USE_NVTX
#include "nvToolsExt.h"
scoped_profile::scoped_profile(const char* name)
{
    nvtxRangePushA(name);
}
scoped_profile::scoped_profile(const char* name, const char* func)
{
    std::stringstream ss;
    ss << name;
    ss << "[";
    ss << func;
    ss << "]";
    nvtxRangePushA(ss.str().c_str());
}
scoped_profile::~scoped_profile()
{
    nvtxRangePop();
}
#else
scoped_profile::scoped_profile(const char* name)
{
}
scoped_profile::scoped_profile(const char* name, const char* func)
{
    
}
scoped_profile::~scoped_profile()
{
    
}



#endif