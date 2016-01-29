#include "Plugins.h"
#include <EagleLib/rcc/ObjectManager.h>
#include <boost/log/trivial.hpp>
#include <EagleLib/Defs.hpp>
#include <EagleLib/Project_defs.hpp>
#ifdef _MSC_VER
#include "Windows.h"

bool CV_EXPORTS EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    static int projectCount = 0;
    BOOST_LOG_TRIVIAL(info) << "Loading plugin " << fullPluginPath;
	HMODULE handle = LoadLibrary(fullPluginPath.c_str());
	if (handle == nullptr)
	{
		auto err = GetLastError();
        BOOST_LOG_TRIVIAL(error) << "Failed to load library due to: " << err;
		return false;
	}
	typedef int(*BuildLevelFunctor)();
	BuildLevelFunctor buildLevel = (BuildLevelFunctor)GetProcAddress(handle, "GetBuildType");
	if (buildLevel)
	{
		if (buildLevel() != BUILD_TYPE)
		{
			BOOST_LOG_TRIVIAL(info) << "Library debug level does not match";
			return false;
		}
	}
	else
	{
		BOOST_LOG_TRIVIAL(warning) << "Build level not defined in library, attempting to load anyways";
	}

	typedef void(*includeFunctor)();
	includeFunctor functor = (includeFunctor)GetProcAddress(handle, "SetupIncludes");
	if (functor)
		functor();
	else
		BOOST_LOG_TRIVIAL(warning) << "Setup Includes not found in plugin " << fullPluginPath;
        
	typedef IPerModuleInterface* (*moduleFunctor)();
	moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetPerModuleInterface");
    if (module)
    {
        auto moduleInterface = module();
        ObjectManager::Instance().setupModule(moduleInterface);
    }
		
	else
	{
		BOOST_LOG_TRIVIAL(warning) << "GetPerModuleInterface not found in plugin " << fullPluginPath;
		module = (moduleFunctor)GetProcAddress(handle, "GetModule");
		if (module)
		{
            auto moduleInterface = module();
			ObjectManager::Instance().setupModule(moduleInterface);
		}
		else
		{
			BOOST_LOG_TRIVIAL(warning) << "GetModule not found in plugin " << fullPluginPath;
			FreeLibrary(handle);
		}	
	}
    return true;
}
#else
#include "dlfcn.h"

bool CV_EXPORTS EagleLib::loadPlugin(const std::string& fullPluginPath)
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
