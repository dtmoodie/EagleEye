#include "Plugins.h"
#include "Manager.h"


#ifdef _MSC_VER
#include "Windows.h"

void EagleLib::loadPlugin(const std::string& fullPluginPath)
{
	HMODULE handle = LoadLibrary(fullPluginPath.c_str());
	if (handle == nullptr)
		return;
	typedef IPerModuleInterface* (*moduleFunctor)();
	moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetModule");
	NodeManager::getInstance().setupModule(module());
	FreeLibrary(handle);
}



#else
#include "dlfcn.h"

void EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    void* handle = dlopen(fullPluginPath.c_str(), RTLD_LAZY);
    typedef IPerModuleInterface* (*moduleFunctor)();

    moduleFunctor module = (moduleFunctor)dlsym(handle, "GetModule");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
           std::cerr << "Cannot load symbol 'GetModule': " << dlsym_error <<
               '\n';
           dlclose(handle);
           return;
       }
    if(module == nullptr)
        return;
    NodeManager::getInstance().setupModule(module());
}



#endif
