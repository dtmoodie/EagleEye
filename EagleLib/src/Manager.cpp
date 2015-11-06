#include "Manager.h"
#include <RuntimeObjectSystem.h>
#include "StdioLogSystem.h"
#include <boost/filesystem.hpp>
#include "nodes/Node.h"
#include "plotters/Plotter.h"
#include <stdarg.h>
#include <assert.h>
#include <iostream>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include "plotters/Plotter.h"
#include <boost/log/trivial.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include "SystemTable.hpp"
#include "remotery/lib/Remotery.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

//#include <IObjectUtils.h>
using namespace EagleLib;

#ifdef _WIN32
    #include "Windows.h"
    #pragma warning( disable : 4996 4800 )
#endif

void CompileLogger::log(int level, const char *format, va_list args)
{
    vsnprintf(m_buff, LOGSYSTEM_MAX_BUFFER-1, format, args);
    // Make sure there's a limit to the amount of rubbish we can output
    m_buff[LOGSYSTEM_MAX_BUFFER-1] = '\0';
    if(callback)
        callback(std::string(m_buff), level);
	switch (level)
	{
	case 0:
	{
		LOG_TRIVIAL(info) << "[RCC] " << m_buff;
		break;
	}
	case 1:
	{
		LOG_TRIVIAL(warning) << "[RCC] " << m_buff;
		break;
	}
	case 2:
	{
		LOG_TRIVIAL(error) << "[RCC] " << m_buff;
		break;
	}
	}

	
#ifdef _WIN32
    OutputDebugStringA( m_buff );
#endif
}

void CompileLogger::LogError(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    log(2, format, args);
}

void CompileLogger::LogWarning(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    log(1, format, args);
	
}

void CompileLogger::LogInfo(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    log(0, format, args);
}
bool TestCallback::TestBuildCallback(const char* file, TestBuildResult type)
{
    bool success = true;
    switch(type)
    {
        case TESTBUILDRRESULT_SUCCESS:
            std::cout << "TESTBUILDRRESULT_SUCCESS - ";
            break;
        case TESTBUILDRRESULT_NO_FILES_TO_BUILD:
            std::cout << "TESTBUILDRRESULT_NO_FILES_TO_BUILD - ";
            success = false;
            break;
        case TESTBUILDRRESULT_BUILD_FILE_GONE:
            std::cout << "TESTBUILDRRESULT_BUILD_FILE_GONE - ";
            success = false;
            break;
        case TESTBUILDRRESULT_BUILD_NOT_STARTED:
            std::cout << "TESTBUILDRRESULT_BUILD_NOT_STARTED - ";
            success = false;
            break;
        case TESTBUILDRRESULT_BUILD_FAILED:
            std::cout << "TESTBUILDRRESULT_BUILD_FAILED - ";
            success = false;
            break;
        case TESTBUILDRRESULT_OBJECT_SWAP_FAIL:
            std::cout << "TESTBUILDRRESULT_OBJECT_SWAP_FAIL - ";
            success = false;
            break;
    }
    std::cout << file << std::endl;
    return success;
}

bool TestCallback::TestBuildWaitAndUpdate()
{
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    return true;
}
UIThreadCallback::UIThreadCallback()
{

}

UIThreadCallback::~UIThreadCallback()
{

}
UIThreadCallback& UIThreadCallback::getInstance()
{
    static UIThreadCallback instance;
    return instance;
}
void UIThreadCallback::addCallback(boost::function<void(void)> f)
{
    queue.push(f);
    if(notifier)
        notifier();
}

void UIThreadCallback::processCallback()
{
    boost::function<void(void)> f;
    if(queue.try_pop(f))
        f();
}
void UIThreadCallback::processAllCallbacks()
{
	LOG_TRACE;
    boost::function<void(void)> f;
    while(queue.try_pop(f))
    {
        f();
    }
}
void UIThreadCallback::clearCallbacks()
{
	LOG_TRACE;
	
    queue.clear();
}

void UIThreadCallback::setUINotifier(boost::function<void(void)> f)
{
	LOG_TRACE;
    notifier = f;
}
boost::asio::io_service ProcessingThreadCallback::service;

boost::asio::io_service& ProcessingThreadCallback::Instance()
{
	return service;
}

void ProcessingThreadCallback::Run()
{
	LOG_TRACE;
	service.run();
}

PlotManager& PlotManager::getInstance()
{
    static PlotManager instance;
    return instance;
}

shared_ptr<Plotter> PlotManager::getPlot(const std::string& plotName)
{
	LOG_TRACE;
    IObjectConstructor* pConstructor = NodeManager::getInstance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(plotName.c_str());
    if(pConstructor && pConstructor->GetInterfaceId() == IID_Plotter)
    {
        IObject* obj = pConstructor->Construct();
        if(obj)
        {
            obj = obj->GetInterface(IID_Plotter);
            if(obj)
            {
                Plotter* plotter = static_cast<Plotter*>(obj);
				if (plotter)
				{
					LOG_TRIVIAL(info) << "[ PlotManager ] successfully generating plot " << plotName;
					return shared_ptr<Plotter>(plotter);
				}
				else
				{
					LOG_TRIVIAL(warning) << "[ PlotManager ] failed to cast to plotter object " << plotName;
				}
			}
			else
			{
				LOG_TRIVIAL(warning) << "[ PlotManager ] incorrect interface " << plotName;
			}
		}
		else
		{
			LOG_TRIVIAL(warning) << "[ PlotManager ] failed to construct plot " << plotName;
		}
	}
	else
	{
		LOG_TRIVIAL(warning) << "[ PlotManager ] failed to get constructor " << plotName;
	}
	return shared_ptr<Plotter>();
}

std::vector<std::string> PlotManager::getAvailablePlots()
{
	LOG_TRACE;
    AUDynArray<IObjectConstructor*> constructors;
    NodeManager::getInstance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    std::vector<std::string> output;
    for(size_t i = 0; i < constructors.Size(); ++i)
    {
        if(constructors[i]->GetInterfaceId() == IID_Plotter)
            output.push_back(constructors[i]->GetName());
    }
    return output;
}

NodeManager& NodeManager::getInstance()
{
	static NodeManager instance;
	return instance;
}

NodeManager::NodeManager()
{
    Init();
}

NodeManager::~NodeManager()
{

}
void NodeManager::addIncludeDir(const std::string& dir, unsigned short projId)
{
	LOG_TRACE << " " << dir;
    m_pRuntimeObjectSystem->AddIncludeDir(dir.c_str(), projId);
}
void NodeManager::addIncludeDirs(const std::string& dirs, unsigned short projId)
{
	if (!dirs.size())
		return;
	boost::char_separator<char> sep("+");
	boost::tokenizer<boost::char_separator<char>> tokens(dirs, sep);
	BOOST_FOREACH(const std::string& t, tokens)
	{
		addIncludeDir(t, projId);
	}
}
void NodeManager::addLinkDirs(const std::string& dirs, unsigned short projId)
{
	LOG_TRACE;
	if (!dirs.size())
		return;
	boost::char_separator<char> sep("+");
	boost::tokenizer<boost::char_separator<char>> tokens(dirs, sep);
	BOOST_FOREACH(const std::string& t, tokens)
	{
		addLinkDir(t, projId);
	}
}
void NodeManager::addLinkDir(const std::string& dir, unsigned short projId)
{
	LOG_TRACE << dir;
	m_pRuntimeObjectSystem->AddLibraryDir(dir.c_str(), projId);
}
void NodeManager::addDefinitions(const std::string& defs, unsigned short projId)
{
	LOG_TRACE;
	if (!defs.size())
		return;
	boost::char_separator<char> sep("+");
	boost::tokenizer<boost::char_separator<char>> tokens(defs, sep);
	BOOST_FOREACH(const std::string& t, tokens)
	{
		m_pRuntimeObjectSystem->SetAdditionalCompileOptions(t.c_str(), projId);
	}
}
int NodeManager::parseProjectConfig(const std::string& file)
{
	if (file.size())
	{
        boost::filesystem::path pfile(file);
        std::string root = pfile.filename().string();
        std::string projectName = root.substr(0, root.size() - 11);

		int id = m_pRuntimeObjectSystem->ParseConfigFile(file.c_str());
        if (id != -1)
        {
            m_projectNames[id] = projectName;
            return id;
        }
	}
	return 0;
}


std::vector<std::string> NodeManager::getLinkDirs(unsigned short projId)
{
    std::vector<std::string> output;
    auto inc = m_pRuntimeObjectSystem->GetLinkDirList(projId);
    for(int i = 0; i < inc.size(); ++i)
    {
        output.push_back(inc[i].m_string);
    }
    return output;
}

std::vector<std::string> NodeManager::getIncludeDirs(unsigned short projId)
{
	LOG_TRACE;
    std::vector<std::string> output;
    auto inc = m_pRuntimeObjectSystem->GetIncludeDirList(projId);
    for(int i = 0; i < inc.size(); ++i)
    {
        output.push_back(inc[i].m_string);
    }
    return output;
}


void NodeManager::addSourceFile(const std::string &file)
{
	LOG_TRACE << " " << file;
    m_pRuntimeObjectSystem->AddToRuntimeFileList(file.c_str());
}

bool
NodeManager::Init()
{
    testCallback = nullptr;
    m_pRuntimeObjectSystem.reset(new RuntimeObjectSystem);
    m_pCompileLogger.reset(new CompileLogger());
	m_systemTable.reset(new SystemTable());
    m_pRuntimeObjectSystem->Initialise(m_pCompileLogger.get(), m_systemTable.get());
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->AddListener(this);
    boost::filesystem::path workingDir(__FILE__);
    std::string includePath = workingDir.parent_path().parent_path().string();
	//m_pRuntimeObjectSystem->SetAdditionalLinkOptions(" -DPARAMETERS_NO_UI ");
	m_pRuntimeObjectSystem->SetAdditionalCompileOptions(" -DPARAMTERS_NO_UI ");
#ifdef _MSC_VER
	
#else
    m_pRuntimeObjectSystem->SetAdditionalCompileOptions("-std=c++11");
#endif // _MSC_VER

#ifdef _DEBUG
    m_pRuntimeObjectSystem->SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_DEBUG);
#else
    m_pRuntimeObjectSystem->SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_PERF);
#endif // _DEBUG
#ifdef _MSC_VER
    includePath += "\\include";
#else
	includePath += "/include";
#endif
    m_pRuntimeObjectSystem->AddIncludeDir(includePath.c_str());
#ifdef NVCC_PATH
	m_pRuntimeObjectSystem->SetCompilerLocation( NVCC_PATH );
#endif
	cv::cuda::GpuMat mat(10, 10, CV_32F);

	Remotery* rmt;
	rmt_CreateGlobalInstance(&rmt);

	CUcontext ctx;
	cuCtxGetCurrent(&ctx);

	rmtCUDABind bind;
	bind.context = ctx;
    bind.CtxSetCurrent = (void*)&cuCtxSetCurrent;
    bind.CtxGetCurrent = (void*)&cuCtxGetCurrent;
    bind.EventCreate = (void*)&cuEventCreate;
    bind.EventDestroy = (void*)&cuEventDestroy;
    bind.EventRecord = (void*)&cuEventRecord;
    bind.EventQuery = (void*)&cuEventQuery;
    bind.EventElapsedTime = (void*)&cuEventElapsedTime;
	rmt_BindCUDA(&bind);
	return true;
}
void NodeManager::RegisterConstructorAddedCallback(boost::function<void(void)> f)
{
    if(f)
        onConstructorsAddedCallbacks.push_back(f);
}

bool
NodeManager::MainLoop()
{
	LOG_TRACE;
	return true;
}
std::vector<std::pair<std::string, int>> NodeManager::getObjectList()
{
	LOG_TRACE;
    std::vector<std::pair<std::string, int>> output;
    AUDynArray<IObjectConstructor*> constructors;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        output.push_back(std::make_pair(std::string(constructors[i]->GetName()), int(constructors[i]->GetProjectId())));
    }
    return output;
}
int NodeManager::getProjectCount()
{
    return    m_pRuntimeObjectSystem->GetProjectCount();
}
std::string NodeManager::getProjectName(int idx)
{
    auto itr = m_projectNames.find(idx);
    if (itr != m_projectNames.end())
    {
        return itr->second;
    }
    return boost::lexical_cast<std::string>(idx);
}

std::vector<std::string> NodeManager::getLinkDependencies(const std::string& objectName)
{
	LOG_TRACE;
    IObjectConstructor* constructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(objectName.c_str());
    std::vector<std::string> linkDependency;
    if(constructor)
    {
        int linkLibCount = constructor->GetMaxNumLinkLibraries();
        linkDependency.reserve(linkLibCount);
        for(int i = 0; i < linkLibCount; ++i)
        {
            const char* lib = constructor->GetLinkLibrary(i);
            if(lib)
                linkDependency.push_back(std::string(lib));
        }
    }
    return linkDependency;
}

void
NodeManager::OnConstructorsAdded()
{
	LOG_TRACE;
	AUDynArray<IObjectConstructor*> constructors;
	m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	std::vector<Node*> newNodes;
    for (size_t i = 0; i < constructors.Size(); ++i)
	{
        size_t numObjects = constructors[i]->GetNumberConstructedObjects();
        for (size_t j = 0; j < numObjects; ++j)
		{
			auto ptr = constructors[i]->GetConstructedObject(j);
            if(ptr)
            {
                ptr = ptr->GetInterface(IID_NodeObject);
                if (ptr)
                {
                    auto nodePtr = static_cast<Node*>(ptr);
                    newNodes.push_back(nodePtr);
                }
            }

		}
	}
    for (size_t i = 0; i < newNodes.size(); ++i)
	{
        for (size_t j = 0; j < newNodes[i]->parameters.size(); ++j)
		{
			if (newNodes[i]->parameters[j]->type & Parameters::Parameter::Input)
			{
				auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(newNodes[i]->parameters[j]);
				inputParam->SetInput(std::string());
			}
		}		
	}
    for(int i = 0; i < onConstructorsAddedCallbacks.size(); ++i)
    {
        onConstructorsAddedCallbacks[i]();
    }
}

shared_ptr<Node> NodeManager::addNode(const std::string &nodeName)
{
	LOG_TRACE << nodeName;
    IObjectConstructor* pConstructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());

    if(pConstructor && pConstructor->GetInterfaceId() == IID_NodeObject)
    {
        IObject* pObj = pConstructor->Construct();
        IObject* interface = pObj->GetInterface(IID_NodeObject);

        if(interface)
        {
            Node* node = static_cast<Node*>(interface);
			try
			{
				node->Init(true);
			}
			catch (cv::Exception &e)
			{
				BOOST_LOG_TRIVIAL(error) << "Failed to initialize node " << nodeName << " due to: " << e.what();
				return shared_ptr<Node>();
			}
			catch (...)
			{
				BOOST_LOG_TRIVIAL(error) << "Failed to initialize node " << nodeName;
				return shared_ptr<Node>();
			}
            
            nodes.push_back(weak_ptr<Node>(node));
            return Node::Ptr(node);
        }else
        {
			BOOST_LOG_TRIVIAL(warning) << "[ NodeManager ] " << nodeName << " not a node";
            // Input nodename is a compatible object but it is not a node
            return shared_ptr<Node>();
        }
    }else
    {
		BOOST_LOG_TRIVIAL(warning) << "[ NodeManager ] " << nodeName << " not a valid node name";
        return shared_ptr<Node>();
    }

    return shared_ptr<Node>();
}
std::vector<shared_ptr<Node>> NodeManager::loadNodes(const std::string& saveFile)
{
	LOG_TRACE;
    boost::filesystem::path path(saveFile);
    if(!boost::filesystem::is_regular_file(path))
    {
        //std::cout << "Unable to load " << saveFile << " doesn't exist, or is not a regular file" << std::endl;
		LOG_TRIVIAL(warning) << "[ NodeManager ] " << saveFile << " doesn't exist or not a regular file";
    }
    cv::FileStorage fs;
    try
    {
        fs.open(saveFile, cv::FileStorage::READ);
    }catch(cv::Exception &e)
    {
        //std::cout << e.what() << std::endl;
		LOG_TRIVIAL(error) << "[ NodeManager ] " << e.what();
    }

    int nodeCount = (int)fs["TopLevelNodeCount"];
	LOG_TRIVIAL(info) << "[ NodeManager ] " << "Loading " << nodeCount << " nodes";
    std::vector<shared_ptr<Node>> nodes;
    nodes.reserve(nodeCount);
    for(int i = 0; i < nodeCount; ++i)
    {
        auto nodeFS = fs["Node-" + boost::lexical_cast<std::string>(i)];
        std::string name = (std::string)nodeFS["NodeName"];
        Node::Ptr node = addNode(name);
        node->Init(nodeFS);
        nodes.push_back(node);
    }
    return nodes;
}

void NodeManager::saveNodes(std::vector<shared_ptr<Node>>& topLevelNodes, const std::string& fileName)
{
    cv::FileStorage fs;
    fs.open(fileName, cv::FileStorage::WRITE);
    saveNodes(topLevelNodes, fs);
    fs.release();
}
void NodeManager::saveNodes(std::vector<shared_ptr<Node>>& topLevelNodes, cv::FileStorage fs)
{
	LOG_TRACE;
    fs << "TopLevelNodeCount" << (int)topLevelNodes.size();

    for(size_t i = 0; i < topLevelNodes.size(); ++i)
    {
        fs << "Node-" + boost::lexical_cast<std::string>(i) << "{";
        topLevelNodes[i]->Serialize(fs);
        fs << "}";
    }
}

bool NodeManager::removeNode(const std::string& nodeName)
{
	LOG_TRACE;
    return false;
}
std::string NodeManager::getNodeFile(const ObjectId& id)
{
	LOG_TRACE;
    AUDynArray<IObjectConstructor*> constructors;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    if(constructors.Size() > id.m_ConstructorId)
    {
        return std::string(constructors[id.m_ConstructorId]->GetFileName());
    }
    return std::string();
}

bool NodeManager::removeNode(ObjectId oid)
{
	LOG_TRACE;
	return false;
}

void 
NodeManager::addConstructors(IAUDynArray<IObjectConstructor*> & constructors)
{
	LOG_TRACE;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->AddConstructors(constructors);
}
void 
NodeManager::setupModule(IPerModuleInterface* pPerModuleInterface)
{
	LOG_TRACE;
	m_pRuntimeObjectSystem->SetupObjectConstructors(pPerModuleInterface);
}
#ifdef _MSC_VER


#else
#include "dlfcn.h"
bool
NodeManager::loadModule(const std::string &filePath)
{
    void* handle = dlopen(filePath.c_str(), RTLD_LAZY);

    typedef IPerModuleInterface* (*moduleFunctor)();

    moduleFunctor module = (moduleFunctor)dlsym(handle, "GetModule");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
           std::cerr << "Cannot load symbol 'GetModule': " << dlsym_error << '\n';
           dlclose(handle);
           return false;
       }
    if(module == nullptr)
        return false;
    NodeManager::getInstance().setupModule(module());

    // Get additional compilation path directories

    typedef void (*SetupModuleFunctor)(IRuntimeObjectSystem*);

    SetupModuleFunctor pathFunctor = (SetupModuleFunctor)dlsym(handle, "SetupModule");
    dlsym_error = dlerror();
    if(dlsym_error)
    {
        std::cerr << "Cannot load symbol 'SetupModule': " << dlsym_error << '\n';
        dlclose(handle);
        return true;
    }
    if(pathFunctor == nullptr)
        return true;
    // This is scary and dangerous :/
    pathFunctor(m_pRuntimeObjectSystem.get());
    dlclose(handle);
    return true;
}
#endif

bool 
NodeManager::CheckRecompile()
{
	LOG_TRACE;
	static boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
	boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
	boost::posix_time::time_duration delta = currentTime - prevTime;
    // Prevent checking too often
    if(delta.total_milliseconds() < 10)
        return false;
    prevTime = currentTime;
    if( m_pRuntimeObjectSystem->GetIsCompiledComplete())
    {
        m_pRuntimeObjectSystem->LoadCompiledModule();
    }
    if(m_pRuntimeObjectSystem->GetIsCompiling())
    {
       return true;
    }else
    {
        m_pRuntimeObjectSystem->GetFileChangeNotifier()->Update(float(delta.total_milliseconds())/1000.0);
    }
    return false;
}


void NodeManager::saveTree(const std::string &fileName)
{
	LOG_TRACE;
}

bool
NodeManager::CheckRecompile(bool swapAllowed)
{
	LOG_TRACE;
    static boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    // Prevent checking too often
    if(delta.total_milliseconds() < 10)
        return false;
    prevTime = currentTime;
    if( m_pRuntimeObjectSystem->GetIsCompiledComplete() && swapAllowed)
    {
        m_pRuntimeObjectSystem->LoadCompiledModule();
    }
    if(m_pRuntimeObjectSystem->GetIsCompiling())
    {
       return true;
    }else
    {
        m_pRuntimeObjectSystem->GetFileChangeNotifier()->Update(float(delta.total_milliseconds())/1000.0);
    }
    return false;
}
bool NodeManager::TestRuntimeCompilation()
{
	LOG_TRACE;
    if(testCallback == nullptr)
        testCallback = new TestCallback();
    m_pRuntimeObjectSystem->TestBuildAllRuntimeHeaders(testCallback,true);
    m_pRuntimeObjectSystem->TestBuildAllRuntimeSourceFiles(testCallback, true);
	return true;
}

void
NodeManager::onNodeRecompile(Node *node)
{
}

Node*
NodeManager::getNode(const ObjectId& id)
{
	LOG_TRACE;
    AUDynArray<IObjectConstructor*> constructors;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    if(!id.IsValid())
        return nullptr;
    if(id.m_ConstructorId >= constructors.Size())
        return nullptr;
    if(id.m_PerTypeId >= constructors[id.m_ConstructorId]->GetNumberConstructedObjects())
        return nullptr;
    IObject* pObj = constructors[id.m_ConstructorId]->GetConstructedObject(id.m_PerTypeId);
    if(!pObj)
        return nullptr;
    pObj = pObj->GetInterface(IID_NodeObject);
    if(!pObj)
        return nullptr;
    return static_cast<Node*>(pObj);
}

Node*
NodeManager::getNode(const std::string &treeName)
{
	LOG_TRACE;
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        if(nodes[i] != nullptr)
        {
            if(nodes[i]->fullTreeName == treeName)
            {
                return nodes[i].get();
            }
        }
    }
    return nullptr;
}

void 
NodeManager::updateTreeName(Node* node, const std::string& prevTreeName)
{
	LOG_TRACE;
    /*m_nodeTree.put(t_nodeTree::path_type{ node->fullTreeName, '.' }, node);
    m_nodeTree.erase(prevTreeName);*/
}

void 
NodeManager::addParameters(Node* node)
{
	LOG_TRACE;
    for (size_t i = 0; i < node->parameters.size(); ++i)
	{
		
	}
}

Parameters::Parameter::Ptr
NodeManager::getParameter(const std::string& name)
{
	LOG_TRACE;
	// Strip off the path for the node
	auto idx = name.find(':');
	std::string parameterName = name.substr(idx+1);
	auto node = getNode(name.substr(0, idx));
    if(node == nullptr)
        return Parameters::Parameter::Ptr();
	return node->getParameter(parameterName);
}

void
NodeManager::getSiblingNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	LOG_TRACE;
}
void NodeManager::setCompileCallback(boost::function<void (const std::string &, int)> &f)
{
	LOG_TRACE;
    m_pCompileLogger->callback = f;
}
void printTreeHelper(std::stringstream& tree, int level, Node* node)
{
	LOG_TRACE;
    for(int i = 0; i < level; ++i)
    {
        tree << "+";
    }
    tree << node->fullTreeName << std::endl;
    for(size_t i = 0; i < node->children.size(); ++i)
    {
        printTreeHelper(tree, level+1, node->children[i].get());
    }
}

void NodeManager::printNodeTree(boost::function<void(std::string)> f)
{
	LOG_TRACE;
    std::stringstream tree;
    std::vector<weak_ptr<Node>> parentNodes;
    // First get the top level nodes for the tree
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        if(nodes[i] != nullptr)
        {
            if(nodes[i]->parent == nullptr)
            {
                parentNodes.push_back(nodes[i]);
            }
        }
    }
    for(size_t i = 0; i < parentNodes.size(); ++i)
    {
        printTreeHelper(tree, 0, parentNodes[i].get());
    }
    if(f)
    {
        f(tree.str());
    }else
    {
        std::cout << tree.str() << std::endl;
    }
}

Node*
NodeManager::getParent(const std::string& sourceNode)
{
	LOG_TRACE;
    return nullptr;
}
void NodeManager::getParentNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	LOG_TRACE;
}

void NodeManager::getAccessibleNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	LOG_TRACE;
	getSiblingNodes(sourceNode, output);
	getParentNodes(sourceNode, output);
}
std::vector<std::string>
NodeManager::getConstructableNodes()
{
	LOG_TRACE;
    AUDynArray<IObjectConstructor*> constructors;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    std::vector<std::string> output;
    for(size_t i = 0; i < constructors.Size(); ++i)
    {
		if (constructors[i])
		{
			if (constructors[i]->GetInterfaceId() == IID_NodeObject)
				output.push_back(constructors[i]->GetName());
		}
		else
		{
			std::cout << "Null constructor idx " << i << std::endl;
		}
    }
    return output;
}
RCppOptimizationLevel NodeManager::getOptimizationLevel()
{
	LOG_TRACE;
    return m_pRuntimeObjectSystem->GetOptimizationLevel();
}

void NodeManager::setOptimizationLevel(RCppOptimizationLevel level)
{
	LOG_TRACE;
    m_pRuntimeObjectSystem->SetOptimizationLevel(level);
}
int NodeManager::getNumLoadedModules()
{
    return m_pRuntimeObjectSystem->GetNumberLoadedModules();
}
std::vector<std::string> NodeManager::getParametersOfType(boost::function<bool(Loki::TypeInfo)> selector)
{
	LOG_TRACE;
    std::vector<std::string> parameters;
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        for(size_t j = 0; j < nodes[i]->parameters.size(); ++j)
        {
            if(selector(nodes[i]->parameters[j]->GetTypeInfo()))
                parameters.push_back(nodes[i]->parameters[j]->GetTreeName());
        }
    }
    return parameters;
}
