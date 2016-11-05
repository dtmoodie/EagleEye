#include "EagleLib/Plugins.h"

#include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include "MetaObject/MetaObjectFactory.hpp"
#include "EagleLib/Detail/Export.hpp"
#include <EagleLib/Project_defs.hpp>
#include "EagleLib/rcc/SystemTable.hpp"
std::vector<std::string> plugins;
std::vector<std::string> EagleLib::ListLoadedPlugins()
{
  return plugins;
}

#ifdef _MSC_VER
#include "Windows.h"
std::string GetLastErrorAsString()
{
    //Get the error message, if any.
    DWORD errorMessageID = ::GetLastError();
    if(errorMessageID == 0)
        return std::string(); //No error message has been recorded

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

    std::string message(messageBuffer, size);

    //Free the buffer.
    LocalFree(messageBuffer);

    return message;
}

//std::vector<std::string> plugins;
//std::vector<std::string> EagleLib::ListLoadedPlugins()
//{
//    return plugins;
//}

bool EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    static int projectCount = 0;
    LOG(info) << "Loading plugin " << fullPluginPath;
    if(!boost::filesystem::is_regular_file(fullPluginPath))
    {
        return false;
    }
    std::string plugin_name = boost::filesystem::path(fullPluginPath).stem().string();
    HMODULE handle = LoadLibrary(fullPluginPath.c_str());
    if (handle == nullptr)
    {
        auto err = GetLastError();
        LOG(debug) << "Failed to load " << plugin_name <<  " due to: [" << err << "] " << GetLastErrorAsString();
        plugins.push_back(fullPluginPath + " - failed");
        return false;
    }
    typedef int(*BuildLevelFunctor)();
    BuildLevelFunctor buildLevel = (BuildLevelFunctor)GetProcAddress(handle, "GetBuildType");
    if (buildLevel)
    {
        if (buildLevel() != BUILD_TYPE)
        {
            LOG(debug) << "Library debug level does not match";
            plugins.push_back(fullPluginPath + " - failed");
            return false;
        }
    }
    else
    {
        LOG(debug) << "Build level not defined in library, attempting to load anyways";
    }

    typedef void(*includeFunctor)();
    includeFunctor functor = (includeFunctor)GetProcAddress(handle, "SetupIncludes");
    if (functor)
        functor();
    else
        LOG(debug) << "Setup Includes not found in plugin " << plugin_name;
        
    typedef IPerModuleInterface* (*moduleFunctor)();
    moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetPerModuleInterface");
    if (module)
    {
        auto moduleInterface = module();
        //ObjectManager::Instance().setupModule(moduleInterface);
        mo::MetaObjectFactory::Instance()->SetupObjectConstructors(moduleInterface);
    }
        
    else
    {
        LOG(debug) << "GetPerModuleInterface not found in plugin " << plugin_name;
        module = (moduleFunctor)GetProcAddress(handle, "GetModule");
        if (module)
        {
            auto moduleInterface = module();
            mo::MetaObjectFactory::Instance()->SetupObjectConstructors(moduleInterface);
        }
        else
        {
            LOG(debug) << "GetModule not found in plugin " << plugin_name;
            FreeLibrary(handle);
        }    
    }
    plugins.push_back(plugin_name + " - success");
    return true;
}
#else
#include "dlfcn.h"

bool EagleLib::loadPlugin(const std::string& fullPluginPath)
{
    return mo::MetaObjectFactory::Instance()->LoadPlugin(fullPluginPath);
}



#endif
struct init_struct
{
    init_struct()
    {
        mo::MetaObjectFactory::Instance(new SystemTable())->RegisterTranslationUnit();
    }
};
static init_struct g_init;
