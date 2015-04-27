#include "Manager.h"
#include <RuntimeObjectSystem.h>
#include "StdioLogSystem.h"
#include <boost/filesystem.hpp>
#include "nodes/Node.h"
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
/*
INodeTreeLeaf::INodeTreeLeaf(Node* node_, INodeTreeLeaf* parent_):
    parent(parent_)
{    m_node = shared_ptr<Node>(node_);}

INodeTreeLeaf::INodeTreeLeaf(shared_ptr<Node> node_, INodeTreeLeaf *parent_):
    m_node(node_), parent(parent_)
{}

NodeTree::NodeTree()
{

}

void NodeTree::addChild(shared_ptr<Node> node, const std::string& path)
{
    auto itr = path.find_first_of('.');
    if(itr != std::string::npos)
    {
        std::string parentNode = path.substr(0, itr);
        INodeTreeLeaf* parent = topLevelNodes.get<LeafName>()[parentNode];
        parent->addChild(node);
    }else
    {
        topLevelNodes.get<0>().push_back(new NodeTreeLeaf(node));
    }
}

shared_ptr<Node> NodeTree::getNode(const std::string& path)
{

}


NodeTreeLeaf::NodeTreeLeaf(shared_ptr<Node> node_, INodeTreeLeaf* parent_):
    INodeTreeLeaf(node_, parent_)
{}

NodeTreeLeaf::NodeTreeLeaf(Node* node_, INodeTreeLeaf* parent_):
    INodeTreeLeaf(node_, parent_)
{
}

void NodeTreeLeaf::addChild(shared_ptr<Node> node_)
{
    if(node_ == nullptr)
        return;
    int count = children.get<LeafName>().count(node_->nodeName);
    node_->treeName = node_->nodeName + "-" + boost::lexical_cast<std::string>(count);
    // Push it into the container based on the insertion order
    children.get<0>().push_back(new NodeTreeLeaf(node_, this));
}
virtual void addChild(Node* node_)
{
    if(node_ == nullptr)
        return;
    int count = children.get<LeafName>().count(node_->nodeName);
    node_->treeName = node_->nodeName + "-" + boost::lexical_cast<std::string>(count);
    // Push it into the container based on the insertion order
    children.get<0>().push_back(new NodeTreeLeaf(node_, this));
}

shared_ptr<Node> NodeTreeLeaf::getNode()
{
    return m_node;
}

shared_ptr<Node> NodeTreeLeaf::getChildNode(const std::string& name)
{
    INodeTreeLeaf* childLeaf = getChildLeaf(name);
    if(childLeaf)
        return childLeaf->getNode();
    return shared_ptr<Node>();
}
shared_ptr<Node> NodeTreeLeaf::getChildNode(int idx)
{
    INodeTreeLeaf* childLeaf = getChildLeaf(idx);
    if(childLeaf)
        return childLeaf->getNode();
    return shared_ptr<Node>();
}

shared_ptr<Node> NodeTreeLeaf::getParentNode()
{
    auto parentLeaf = getParentLeaf();
    if(parentLeaf)
        return parentLeaf->getNode();
    return shared_ptr<Node>();
}

INodeTreeLeaf* NodeTreeLeaf::getParentLeaf()
{
    return parent;
}

INodeTreeLeaf* NodeTreeLeaf::getChildLeaf(const std::string& name)
{
    auto itr = children.get<LeafName>().find(name);
    if(itr != children.get<LeafName>().end())
        return *itr;
    return nullptr;
}

INodeTreeLeaf* NodeTreeLeaf::getChildLeaf(int idx)
{
    if(idx >= children.get<0>().size())
        return nullptr;
    return children.get<0>()[idx];
}

void NodeTreeLeaf::swapChildren(const std::string& currentName, const std::string& newName)
{

}

void NodeTreeLeaf::swapChildren(int initialIdx, int newIdx)
{

}
*/
void CompileLogger::log(int level, const char *format, va_list args)
{
    int result = vsnprintf(m_buff, LOGSYSTEM_MAX_BUFFER-1, format, args);
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
    includePath += "/include";
    m_pRuntimeObjectSystem->AddIncludeDir(includePath.c_str());
	m_pRuntimeObjectSystem->AddIncludeDir(BOOST_INCLUDES);
	m_pRuntimeObjectSystem->AddIncludeDir(OPENCV_INCLUDES);
    m_pRuntimeObjectSystem->AddIncludeDir(CUDA_INCLUDES);

	m_pRuntimeObjectSystem->AddLibraryDir(BOOST_LIB_DIR);
	m_pRuntimeObjectSystem->AddLibraryDir(OPENCV_LIB_DIR);
    m_pRuntimeObjectSystem->AddLibraryDir(CUDA_LIB_DIR);

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
	for (int i = 0; i < constructors.Size(); ++i)
	{
		int numObjects = constructors[i]->GetNumberConstructedObjects();
		for (int j = 0; j < numObjects; ++j)
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
	for (int i = 0; i < newNodes.size(); ++i)
	{
		for (int j = 0; j < newNodes[i]->parameters.size(); ++j)
		{
			newNodes[i]->parameters[j]->setSource(std::string());
		}		
	}
}

shared_ptr<Node> NodeManager::addNode(const std::string &nodeName)
{

    IObjectConstructor* pConstructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());

    if(pConstructor)
    {
        IObject* pObj = pConstructor->Construct();
        IObject* interface = pObj->GetInterface(IID_NodeObject);

        if(interface)
        {
            Node* node = static_cast<Node*>(interface);
            node->Init(true);
            // Add the first node here, however later the tree is updated in updateTreeName
            /*if (m_nodeMap.size() == 0)
                m_nodeTree.put(t_nodeTree::path_type{node->fullTreeName, '.' }, node);*/
            m_nodeMap[nodeName].push_back(node);
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
bool NodeManager::removeNode(const std::string& nodeName)
{

    return false;
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
void writeOutNodes(t_nodeTree& ptree, int level)
{
    for(auto itr = ptree.begin(); itr != ptree.end(); ++itr)
    {
        std::cout << level << ": " << itr->first << std::endl;
        writeOutNodes(itr->second, level + 1);
    }
}

void NodeManager::saveTree(const std::string &fileName)
{
    //writeOutNodes(m_nodeTree, 0);
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

void
NodeManager::onNodeRecompile(Node *node)
{
    deletedNodes.push_back(node);
    deletedNodeIDs.push_back(node->m_OID);
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
    /*Node* ptr = nullptr;
    try
    {
        ptr = m_nodeTree.get<Node*>(treeName);
    }catch(boost::exception &err)
    {
        std::cout << "Error getting node by name: " << treeName << std::endl;
    }*/

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
	for (int i = 0; i < node->parameters.size(); ++i)
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
    /*auto nodePtr = getParent(sourceNode);
	if (!nodePtr)
		return;

	auto node = m_nodeTree.get_child(getParent(sourceNode)->fullTreeName);
	for (auto itr = node.begin(); itr != node.end(); ++itr)
	{
		output.push_back(itr->second.get_value<Node*>());
    }*/
}
void NodeManager::setCompileCallback(boost::function<void (const std::string &, int)> &f)
{
    m_pCompileLogger->callback = f;
}

Node*
NodeManager::getParent(const std::string& sourceNode)
{
    /*auto idx = sourceNode.find_last_of('.');
	if (idx > sourceNode.size())
		return nullptr;
	std::string treeName = sourceNode.substr(0, idx);
    Node* node = nullptr;
    try
    {
        node = m_nodeTree.get<Node*>(treeName);
    }catch(...)
    {
        return nullptr;
    }
    return node;*/
    return nullptr;
}
void NodeManager::getParentNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
    /*Node* parent = getParent(sourceNode);
	if (!parent)
		return;
	output.push_back(parent);
    getParentNodes(parent->fullTreeName, output);*/
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
    for(int i = 0; i < constructors.Size(); ++i)
    {
        output.push_back(constructors[i]->GetName());
    }
    return output;
}
