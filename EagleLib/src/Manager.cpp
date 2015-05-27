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
    std::cout << m_buff;
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
    boost::function<void(void)> f;
    while(queue.try_pop(f))
    {
        f();
    }
}
void UIThreadCallback::setUINotifier(boost::function<void(void)> f)
{
    notifier = f;
}

PlotManager& PlotManager::getInstance()
{
    static PlotManager instance;
    return instance;
}

shared_ptr<Plotter> PlotManager::getPlot(const std::string& plotName)
{
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
					return shared_ptr<Plotter>(plotter);
				}
            }
        }
    }
	return shared_ptr<Plotter>();
}

std::vector<std::string> PlotManager::getAvailablePlots()
{
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

bool
NodeManager::Init()
{
    testCallback = nullptr;
    m_pRuntimeObjectSystem.reset(new RuntimeObjectSystem);
    m_pCompileLogger.reset(new CompileLogger());
    m_pRuntimeObjectSystem->Initialise(m_pCompileLogger.get(), nullptr);
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->AddListener(this);
    boost::filesystem::path workingDir(__FILE__);
    std::string includePath = workingDir.parent_path().parent_path().string();
#ifdef _MSC_VER
	
#else
    m_pRuntimeObjectSystem->SetAdditionalCompileOptions("-std=c++11");
#endif

#ifdef _DEBUG
    m_pRuntimeObjectSystem->SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_DEBUG);
#else
    m_pRuntimeObjectSystem->SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_PERF);
#endif

    includePath += "/include";
    m_pRuntimeObjectSystem->AddIncludeDir(includePath.c_str());
	m_pRuntimeObjectSystem->AddIncludeDir(BOOST_INCLUDES);
	m_pRuntimeObjectSystem->AddIncludeDir(OPENCV_INCLUDES);
    m_pRuntimeObjectSystem->AddIncludeDir(CUDA_INCLUDES);

	m_pRuntimeObjectSystem->AddLibraryDir(BOOST_LIB_DIR);
	m_pRuntimeObjectSystem->AddLibraryDir(OPENCV_LIB_DIR);
    m_pRuntimeObjectSystem->AddLibraryDir(CUDA_LIB_DIR);
#ifdef BINARY_DIR
	m_pRuntimeObjectSystem->AddLibraryDir(BINARY_DIR);
#endif
#ifdef HAVE_PCL
    //m_pRuntimeObjectSystem->AddLibraryDir(PCL_LIB_DIR);
    m_pRuntimeObjectSystem->AddIncludeDir(PCL_INCLUDES);
#endif
	return true;
}

bool
NodeManager::MainLoop()
{
	return true;
}

void
NodeManager::OnConstructorsAdded()
{
	AUDynArray<IObjectConstructor*> constructors;
	m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	std::vector<Node*> newNodes;
    for (size_t i = 0; i < constructors.Size(); ++i)
	{
        size_t numObjects = constructors[i]->GetNumberConstructedObjects();
        for (size_t j = 0; j < numObjects; ++j)
		{
			auto ptr = constructors[i]->GetConstructedObject(j);
			ptr = ptr->GetInterface(IID_NodeObject);
			if (ptr)
			{
                auto nodePtr = static_cast<Node*>(ptr);
                //m_nodeTree.put(t_nodeTree::path_type{nodePtr->fullTreeName,'.'}, nodePtr);
				newNodes.push_back(nodePtr);
			}
		}
	}
    for (size_t i = 0; i < newNodes.size(); ++i)
	{
        for (size_t j = 0; j < newNodes[i]->parameters.size(); ++j)
		{
			newNodes[i]->parameters[j]->setSource(std::string());
		}		
	}
}

shared_ptr<Node> NodeManager::addNode(const std::string &nodeName)
{
    IObjectConstructor* pConstructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());

    if(pConstructor && pConstructor->GetInterfaceId() == IID_NodeObject)
    {
        IObject* pObj = pConstructor->Construct();
        IObject* interface = pObj->GetInterface(IID_NodeObject);

        if(interface)
        {
            Node* node = static_cast<Node*>(interface);
            node->Init(true);
            nodes.push_back(weak_ptr<Node>(node));
            return Node::Ptr(node);
        }else
        {
            // Input nodename is a compatible object but it is not a node
            return shared_ptr<Node>();
        }
    }else
    {
        // Invalid nodeName
        return shared_ptr<Node>();
    }

    return shared_ptr<Node>();
}
std::vector<shared_ptr<Node>> NodeManager::loadNodes(const std::string& saveFile)
{
    cv::FileStorage fs;
    try
    {
        fs.open(saveFile, cv::FileStorage::READ);
    }catch(cv::Exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    int nodeCount = (int)fs["TopLevelNodeCount"];
    std::vector<shared_ptr<Node>> nodes;
    nodes.reserve(nodeCount);
    for(int i = 0;i < nodeCount; ++i)
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

    return false;
}
std::string NodeManager::getNodeFile(const ObjectId& id)
{
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
 /*   Node* node =getNode(oid);
    if(node == nullptr)
        return false;
    auto path = node->fullTreeName;
    // First we need to get the parent property tree node.
    /////// GRRRRR I've tried everything but I can't get the property tree to delete the node.
    saveTree("test.xml");
    auto idx = path.find_last_of(".");
    auto parentPath = path.substr(0, idx);
    auto parentNode = m_nodeTree.get_child(parentPath);
    for(auto it = parentNode.begin(); it != parentNode.end(); ++it)
    {
        if(it->second.get_value<Node*>() == node)
        {
            parentNode.erase(it);
            break;
        }
    }
    saveTree("test.xml");
    delete m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(oid.m_ConstructorId)->GetConstructedObject(oid.m_PerTypeId);
    return true;*/
	return false;
}

void 
NodeManager::addConstructors(IAUDynArray<IObjectConstructor*> & constructors)
{
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->AddConstructors(constructors);
}
void 
NodeManager::setupModule(IPerModuleInterface* pPerModuleInterface)
{
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

}

bool
NodeManager::CheckRecompile(bool swapAllowed)
{
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
    /*m_nodeTree.put(t_nodeTree::path_type{ node->fullTreeName, '.' }, node);
    m_nodeTree.erase(prevTreeName);*/
}

void 
NodeManager::addParameters(Node* node)
{
    for (size_t i = 0; i < node->parameters.size(); ++i)
	{
		
	}
}

boost::shared_ptr< Parameter > 
NodeManager::getParameter(const std::string& name)
{
	// Strip off the path for the node
	auto idx = name.find(':');
	std::string parameterName = name.substr(idx+1);
	auto node = getNode(name.substr(0, idx));
    if(node == nullptr)
        return Parameter::Ptr();
	return node->getParameter(parameterName);
}

void
NodeManager::getSiblingNodes(const std::string& sourceNode, std::vector<Node*>& output)
{

}
void NodeManager::setCompileCallback(boost::function<void (const std::string &, int)> &f)
{
    m_pCompileLogger->callback = f;
}
void printTreeHelper(std::stringstream& tree, int level, Node* node)
{
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

    return nullptr;
}
void NodeManager::getParentNodes(const std::string& sourceNode, std::vector<Node*>& output)
{

}

void NodeManager::getAccessibleNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	getSiblingNodes(sourceNode, output);
	getParentNodes(sourceNode, output);
}
std::vector<std::string>
NodeManager::getConstructableNodes()
{
    AUDynArray<IObjectConstructor*> constructors;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    std::vector<std::string> output;
    for(size_t i = 0; i < constructors.Size(); ++i)
    {
        if(constructors[i]->GetInterfaceId() == IID_NodeObject)
            output.push_back(constructors[i]->GetName());
    }
    return output;
}
RCppOptimizationLevel NodeManager::getOptimizationLevel()
{
    return m_pRuntimeObjectSystem->GetOptimizationLevel();
}

void NodeManager::setOptimizationLevel(RCppOptimizationLevel level)
{
    m_pRuntimeObjectSystem->SetOptimizationLevel(level);
}
int NodeManager::getNumLoadedModules()
{
    return m_pRuntimeObjectSystem->GetNumberLoadedModules();
}
std::vector<std::string> NodeManager::getParametersOfType(boost::function<bool(Loki::TypeInfo&)> selector)
{
    std::vector<std::string> parameters;
    for(size_t i = 0; i < nodes.size(); ++i)
    {
        for(size_t j = 0; j < nodes[i]->parameters.size(); ++j)
        {
            if(selector(nodes[i]->parameters[j]->typeInfo))
                parameters.push_back(nodes[i]->parameters[j]->treeName);
        }
    }
    return parameters;
}
