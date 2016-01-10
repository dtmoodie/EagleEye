#pragma once
#include <EagleLib/Logging.h>
#include "Defs.hpp"
#include "IObjectFactorySystem.h"
#include "IRuntimeObjectSystem.h"
#include <memory>
#include <functional>
#include <map>
#include "shared_ptr.hpp"


struct SystemTable;
struct IRuntimeObjectSystem;
namespace EagleLib
{
	const size_t LOGSYSTEM_MAX_BUFFER = 20000;


	class CompileLogger : public ICompilerLogger
	{
		char m_buff[LOGSYSTEM_MAX_BUFFER];
		void log(int level, const char* format, va_list args);

	public:
		std::function<void(const std::string&, int)> callback;
		virtual void LogError(const char * format, ...);
		virtual void LogWarning(const char * format, ...);
		virtual void LogInfo(const char * format, ...);

	};
	class TestCallback : public ITestBuildNotifier
	{
	public:
		virtual bool TestBuildCallback(const char* file, TestBuildResult type);
		virtual bool TestBuildWaitAndUpdate();
	};

	class EAGLE_EXPORTS ObjectManager : public IObjectFactoryListener
	{
	public:
        template<typename T, int IID> shared_ptr<T> GetObject(const std::string& object_name)
        {
            auto constructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(object_name.c_str());
            if (constructor)
            {
                if (IID == constructor->GetInterfaceId())
                {
                    auto object = constructor->Construct();
                    return shared_ptr<T>(object);
                }
                LOG(warning) << "interface ID (" << IID << " doesn't match constructor interface " << constructor->GetInterfaceId();
            }
            LOG(warning) << "Constructor for " << object_name " doesn't exist";
            return shared_ptr<T>();
        }
		static ObjectManager& Instance();
		void addIncludeDir(const std::string& dir, unsigned short projId = 0);
		void addDefinitions(const std::string& defs, unsigned short projId = 0);
		void addSourceFile(const std::string& file);
		void addIncludeDirs(const std::string& dirs, unsigned short projId = 0);
		void addLinkDir(const std::string& dir, unsigned short projId = 0);
		void addLinkDirs(const std::string& dirs, unsigned short projId = 0);
		int parseProjectConfig(const std::string& file);
		std::vector<std::string> getLinkDirs(unsigned short projId = 0);
		std::vector<std::string> getIncludeDirs(unsigned short projId = 0);
		virtual bool TestRuntimeCompilation();
		RCppOptimizationLevel getOptimizationLevel();
		void setOptimizationLevel(RCppOptimizationLevel level);
		int getNumLoadedModules();
		void setCompileCallback(std::function<void(const std::string&, int)> & f);
		void RegisterConstructorAddedCallback(std::function<void(void)> f);
		virtual bool CheckRecompile(bool swapAllowed = true);
		void setupModule(IPerModuleInterface* pPerModuleInterface);
		std::vector<std::pair<std::string, int>> getObjectList();
		std::vector<std::string> getLinkDependencies(const std::string& objectName);
		int getProjectCount();
		std::string getProjectName(int idx);
		virtual void OnConstructorsAdded();
		void addConstructors(IAUDynArray<IObjectConstructor*> & constructors);
	private:
		ObjectManager();
		friend class PlotManager;
		friend class NodeManager;

		std::shared_ptr<IRuntimeObjectSystem>             m_pRuntimeObjectSystem;
		std::shared_ptr<CompileLogger>                    m_pCompileLogger;
		TestCallback*                                       m_pTestCallback;
		std::shared_ptr<SystemTable>						m_systemTable;
		std::vector<std::function<void(void)>>            onConstructorsAddedCallbacks;
		std::map<int, std::string>                          m_projectNames;
	};// class ObjectManager
}