#include "Plugins.h"
#include <EagleLib/rcc/ObjectManager.h>
#include <boost/log/trivial.hpp>
#include <EagleLib/Defs.hpp>
#include <EagleLib/Project_defs.hpp>
#ifdef _MSC_VER
#include "Windows.h"
std::vector<std::string> plugins;
std::vector<std::string> EagleLib::ListLoadedPlugins()
{
	return plugins;
}

bool EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    static int projectCount = 0;
	LOG(info) << "Loading plugin " << fullPluginPath;
	HMODULE handle = LoadLibrary(fullPluginPath.c_str());
	if (handle == nullptr)
	{
		auto err = GetLastError();
		LOG(error) << "Failed to load library due to: " << err;
		plugins.push_back(fullPluginPath + " - failed");
		return false;
	}
	typedef int(*BuildLevelFunctor)();
	BuildLevelFunctor buildLevel = (BuildLevelFunctor)GetProcAddress(handle, "GetBuildType");
	if (buildLevel)
	{
		if (buildLevel() != BUILD_TYPE)
		{
			LOG(info) << "Library debug level does not match";
			plugins.push_back(fullPluginPath + " - failed");
			return false;
		}
	}
	else
	{
		LOG(warning) << "Build level not defined in library, attempting to load anyways";
	}

	typedef void(*includeFunctor)();
	includeFunctor functor = (includeFunctor)GetProcAddress(handle, "SetupIncludes");
	if (functor)
		functor();
	else
		LOG(warning) << "Setup Includes not found in plugin " << fullPluginPath;
        
	typedef IPerModuleInterface* (*moduleFunctor)();
	moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetPerModuleInterface");
    if (module)
    {
        auto moduleInterface = module();
        ObjectManager::Instance().setupModule(moduleInterface);
    }
		
	else
	{
		LOG(warning) << "GetPerModuleInterface not found in plugin " << fullPluginPath;
		module = (moduleFunctor)GetProcAddress(handle, "GetModule");
		if (module)
		{
            auto moduleInterface = module();
			ObjectManager::Instance().setupModule(moduleInterface);
		}
		else
		{
			LOG(warning) << "GetModule not found in plugin " << fullPluginPath;
			FreeLibrary(handle);
		}	
	}
	plugins.push_back(fullPluginPath + " - success");
    return true;
}
#else
#include "dlfcn.h"

bool EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    void* handle = dlopen(fullPluginPath.c_str(), RTLD_LAZY);
    // Fallback on old module


    typedef IPerModuleInterface* (*moduleFunctor)();

    moduleFunctor module = (moduleFunctor)dlsym(handle, "GetPerModuleInterface");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
           std::cerr << dlsym_error << '\n';
           module = (moduleFunctor)dlsym(handle, "GetModule");
           dlsym_error = dlerror();
           if(dlsym_error)
           {
                std::cerr << dlsym_error << '\n';
                return false;
           }
           std::cout << "Found symbol 'GetModule'" << std::endl;

       }
    if(module == nullptr)
    {
        std::cout << "module == nullptr" << std::endl;
        return false;
    }
    ObjectManager::Instance().setupModule(module());
    typedef void(*includeFunctor)();
    includeFunctor functor = (includeFunctor)dlsym(handle, "SetupIncludes");
    if(functor)
        functor();

    return true;
}



#endif
