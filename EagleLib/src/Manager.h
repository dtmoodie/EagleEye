#pragma once
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define EAGLE_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define EAGLE_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define EAGLE_EXPORTS
#endif



#ifdef __cplusplus
#define SETUP_PROJECT_DEF extern "C"{ EAGLE_EXPORTS void SetupIncludes(); }
#else
#define SETUP_PROJECT_DEF EAGLE_EXPORTS void SetupIncludes();
#endif

#ifdef PROJECT_INCLUDES
	#ifdef PROJECT_LIB_DIRS
		#define SETUP_PROJECT_IMPL void SetupIncludes(){ 																	\
				EagleLib::NodeManager::getInstance().addIncludeDirs(PROJECT_INCLUDES);										\
				EagleLib::NodeManager::getInstance().addLinkDirs(PROJECT_LIB_DIRS);}		
	#else
		#define SETUP_PROJECT_IMPL void SetupIncludes(){EagleLib::NodeManager::getInstance().addIncludeDirs(PROJECT_INCLUDES);}		
	#endif
#else
	#ifdef PROJECT_LIB_DIRS
		#define SETUP_PROJECT_IMPL void SetupIncludes()	{EagleLib::NodeManager::getInstance().addLinkDirs(PROJECT_LIB_DIRS);}
	#else
#ifndef EagleLIB_EXPORTS
        //#pragma message( "Neither PROJECT_LIB_DIRS nor PROJECT_INCLUDES defined" )
#endif
		#define SETUP_PROJECT_IMPL void SetupIncludes() {}
	#endif
#endif


#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H


#include <map>
#include <list>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/asio/io_service.hpp>
#include "ObjectInterface.h"
#include "IObjectFactorySystem.h"
#include "IObject.h"
#include "IRuntimeObjectSystem.h"

#include <opencv2/core/cvdef.h>

#include "LokiTypeInfo.h"

#include "nodes/Node.h"
#include "CudaUtils.hpp"

#include "Parameters.hpp"



namespace EagleLib
{
    class Node;
	class Parameter;
    class Plotter;
    const size_t LOGSYSTEM_MAX_BUFFER = 20000;


    class CompileLogger: public ICompilerLogger
    {
        char m_buff[LOGSYSTEM_MAX_BUFFER];
        void log(int level, const char* format,  va_list args);

    public:
        boost::function<void(const std::string&, int)> callback;
        virtual void LogError(const char * format, ...);
        virtual void LogWarning(const char * format, ...);
        virtual void LogInfo(const char * format, ...);

    };
    class TestCallback: public ITestBuildNotifier
    {
    public:
        virtual bool TestBuildCallback(const char* file, TestBuildResult type);
        virtual bool TestBuildWaitAndUpdate();
    };
	class CV_EXPORTS UIThreadCallback
    {
    private:
        concurrent_queue<boost::function<void(void)>> queue;
        UIThreadCallback();
        boost::function<void(void)> notifier;
        virtual ~UIThreadCallback();
    public:
        static UIThreadCallback& getInstance();
        void addCallback(boost::function<void(void)> f);
        void clearCallbacks();
        void processCallback();
        void processAllCallbacks();
        void setUINotifier(boost::function<void(void)> f);
    };
	class CV_EXPORTS ProcessingThreadCallback
	{
	private:
		static boost::asio::io_service service;

	public:
		static boost::asio::io_service& Instance();
		static void Run();

	};

    class CV_EXPORTS PlotManager
    {
    public:
        static PlotManager& getInstance();
        shared_ptr<Plotter> getPlot(const std::string& plotName);
        std::vector<std::string> getAvailablePlots();
    };

    class CV_EXPORTS NodeManager : public IObjectFactoryListener
    {

	public:
		static NodeManager& getInstance();

        shared_ptr<Node> addNode(const std::string& nodeName);
        std::vector<shared_ptr<Node>> loadNodes(const std::string& saveFile);
        void saveNodes(std::vector<shared_ptr<Node>>& topLevelNodes, const std::string& fileName);
        void saveNodes(std::vector<shared_ptr<Node>>& topLevelNodes, cv::FileStorage fs);
        void printNodeTree(boost::function<void(std::string)> f = boost::function<void(std::string)>());
		void addConstructors(IAUDynArray<IObjectConstructor*> & constructors);
		void setupModule(IPerModuleInterface* pPerModuleInterface);
        bool loadModule(const std::string& filePath);
        void saveTree(const std::string& fileName);
        std::string getNodeFile(const ObjectId& id);
        bool Init();

        bool MainLoop(); // Depricated

        virtual void OnConstructorsAdded();

        virtual bool CheckRecompile();
        virtual bool CheckRecompile(bool swapAllowed);
        virtual bool TestRuntimeCompilation();
        RCppOptimizationLevel getOptimizationLevel();
        void setOptimizationLevel(RCppOptimizationLevel level);
        int getNumLoadedModules();

        void onNodeRecompile(Node* node);

        Node* getNode(const ObjectId& id);
        Node* getNode(const std::string& treeName);
        bool removeNode(const std::string& nodeName);
        bool removeNode(ObjectId oid);

		void updateTreeName(Node* node, const std::string& prevTreeName);
		void addParameters(Node* node);
		Parameters::Parameter::Ptr getParameter(const std::string& name);
        void setCompileCallback(boost::function<void(const std::string&, int)> & f);

		void getSiblingNodes(const std::string& sourceNode, std::vector<Node*>& output);
		void getParentNodes(const std::string& sourceNode, std::vector<Node*>& output);
		void getAccessibleNodes(const std::string& sourceNode, std::vector<Node*>& output);
		Node* getParent(const std::string& sourceNode);
        std::vector<std::string> getConstructableNodes();
        std::vector<std::string> getParametersOfType(boost::function<bool(Loki::TypeInfo)> selector);
        void addIncludeDir(const std::string& dir);
        void addSourceFile(const std::string& file);
		void addIncludeDirs(const std::string& dirs);
		void addLinkDir(const std::string& dir);
		void addLinkDirs(const std::string& dirs);
        std::vector<std::string> getLinkDirs();
        std::vector<std::string> getIncludeDirs();

        std::vector<std::string> getObjectList();
        std::vector<std::string> getLinkDependencies(const std::string& objectName);

        void RegisterConstructorAddedCallback(boost::function<void(void)> f);

	private:
        friend class PlotManager;
		NodeManager();
		virtual ~NodeManager();
        boost::shared_ptr<IRuntimeObjectSystem>             m_pRuntimeObjectSystem;
        boost::shared_ptr<CompileLogger>                    m_pCompileLogger;
        TestCallback*                                       testCallback;
        std::vector<weak_ptr<Node>>                         nodes;
        std::vector<boost::function<void(void)>>             onConstructorsAddedCallbacks;
		
    }; // class NodeManager
} // namespace EagleLib
#endif /*FILE_MANAGER_H*/
