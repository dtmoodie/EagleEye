#include "Plugins.h"
#include "Manager.h"


#ifdef _MSC_VER
#include "Windows.h"

bool EagleLib::loadPlugin(const std::string& fullPluginPath)
{
	HMODULE handle = LoadLibrary(fullPluginPath.c_str());
	if (handle == nullptr)
        return false;
	typedef IPerModuleInterface* (*moduleFunctor)();
	moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetModule");
	NodeManager::getInstance().setupModule(module());
	FreeLibrary(handle);
    return true;
}



#else
#include "dlfcn.h"

bool EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    void* handle = dlopen(fullPluginPath.c_str(), RTLD_LAZY);
    // Fallback on old module


    typedef IPerModuleInterface* (*moduleFunctor)();

    moduleFunctor module = (moduleFunctor)dlsym(handle, "GetModule");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
           std::cerr << "Cannot load symbol 'GetModule': " << dlsym_error <<
               '\n';
           dlclose(handle);
           return false;
       }
    if(module == nullptr)
        return false;
    NodeManager::getInstance().setupModule(module());
    return true;
}



#endif
