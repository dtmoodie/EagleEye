#include "Plugins.h"
#include "Manager.h"
#include <boost/log/trivial.hpp>

#ifdef _MSC_VER
#include "Windows.h"

bool CV_EXPORTS EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    static int projectCount = 0;
	//std::cout << "Loading plugin " << fullPluginPath << std::endl;
    BOOST_LOG_TRIVIAL(info) << "Loading plugin " << fullPluginPath;
	HMODULE handle = LoadLibrary(fullPluginPath.c_str());
	if (handle == nullptr)
	{
		auto err = GetLastError();
		//std::cout << "Failed to load library due to: " << err << std::endl;
        BOOST_LOG_TRIVIAL(error) << "Failed to load library due to: " << err;
		return false;
	}
        
	typedef IPerModuleInterface* (*moduleFunctor)();
	moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetPerModuleInterface");
    if (module)
    {
        auto moduleInterface = module();
        NodeManager::getInstance().setupModule(moduleInterface);
    }
		
	else
	{
		BOOST_LOG_TRIVIAL(warning) << "GetPerModuleInterface not found in plugin " << fullPluginPath;
		module = (moduleFunctor)GetProcAddress(handle, "GetModule");
		if (module)
		{
            auto moduleInterface = module();
            NodeManager::getInstance().setupModule(moduleInterface);
		}
		else
		{
			BOOST_LOG_TRIVIAL(warning) << "GetModule not found in plugin " << fullPluginPath;
			FreeLibrary(handle);
		}
		
	}
	typedef void(*includeFunctor)();
	includeFunctor functor = (includeFunctor)GetProcAddress(handle, "SetupIncludes");
	if (functor)
		functor();
	else
		BOOST_LOG_TRIVIAL(warning) << "Setup Includes not found in plugin " << fullPluginPath;
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
    NodeManager::getInstance().setupModule(module());
    typedef void(*includeFunctor)();
    includeFunctor functor = (includeFunctor)dlsym(handle, "SetupIncludes");
    if(functor)
        functor();

    return true;
}



#endif
