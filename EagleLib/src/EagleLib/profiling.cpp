#include "EagleLib/profiling.h"
#include <string>
#include <sstream>
#ifdef USE_NVTX
//#include "nvToolsExt.h"
#if WIN32
#include "Windows.h"
#else
#include "dlfcn.h"
#endif
using namespace EagleLib;

typedef int(*push_f)(const char*);
typedef int(*pop_f)();
push_f push = NULL;
pop_f pop = NULL;

void EagleLib::InitProfiling()
{
#if WIN32
    HMODULE handle = LoadLibrary("nvToolsExt64_1.lib");
    if(handle)
    {
        push = (push_f)GetProcAddress(handle, "nvtxRangePushA");
        pop = (pop_f)GetProcAddress(handle, "nvtxRangePop");
    }

#else


#endif
}



scoped_profile::scoped_profile(const char* name)
{
    //nvtxRangePushA(name);
    if(push)
        (*push)(name);
}
scoped_profile::scoped_profile(const char* name, const char* func)
{
    if(push)
    {
        std::stringstream ss;
        ss << name;
        ss << "[";
        ss << func;
        ss << "]";
        (*push)(ss.str().c_str());
    }
}
scoped_profile::~scoped_profile()
{
    if(pop)
    {
        (*pop)();
    }
    
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