#include "Plugins.h"
#include "Manager.h"


#ifdef _MSC_VER
#include "Windows.h"

bool CV_EXPORTS EagleLib::loadPlugin(const std::string& fullPluginPath)
{
	
	HMODULE handle = LoadLibrary(fullPluginPath.c_str());
	if (handle == nullptr)
	{
		auto err = GetLastError();
		std::cout << "Failed to load library due to: " << err << std::endl;
		return false;
	}
        
	typedef IPerModuleInterface* (*moduleFunctor)();
	moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetModule");
	if (module)
		NodeManager::getInstance().setupModule(module());
	else
	{
		std::cout << "GetModule not found in plugin " << fullPluginPath << std::endl;
		FreeLibrary(handle);
	}
	typedef void(*includeFunctor)();
	includeFunctor functor = (includeFunctor)GetProcAddress(handle, "SetupIncludes");
	if (functor)
		functor();
    return true;
}



#else
#include "dlfcn.h"

bool CV_EXPORTS EagleLib::loadPlugin(const std::string& fullPluginPath)
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
    typedef void(*includeFunctor)();
    includeFunctor functor = (includeFunctor)dlsym(handle, "SetupIncludes");
    if(functor)
        functor();

    return true;
}



#endif
