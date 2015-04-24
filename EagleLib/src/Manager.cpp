#include "Manager.h"
#include <RuntimeObjectSystem.h>
#include "StdioLogSystem.h"
#include <boost/filesystem.hpp>
#include "nodes/Node.h"
//#include <IObjectUtils.h>
using namespace EagleLib;

/*
 * TODO:
 *  - Nodes can be swapped in and out correctly, but any changes to their parameters is lost.  It would seem..... Fixed by swapping old nodes for the newly created ones.  Will need to figure out how to do this elegantly without leaking memory.
 *  - See if swapping of nodes causes a memory leak.  Might not really matter but need to figoure out how to tell the objectfactory when you delete an object.
 *  - Look into loading constructors from libraries.
 *
 *
 *
 *
*/
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
    m_pCompileLogger.reset(new StdioLogSystem());
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
                m_nodeTree.put(t_nodeTree::path_type{nodePtr->fullTreeName,'.'}, nodePtr);
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

Node* NodeManager::addNode(const std::string &nodeName)
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
            if (m_nodeMap.size() == 0)
				m_nodeTree.put(t_nodeTree::path_type{node->fullTreeName, '.' }, node);
            m_nodeMap[nodeName].push_back(node);
            return node;
        }else
        {
            // Input nodename is a compatible object but it is not a node
            return nullptr;
        }
    }else
    {
        // Invalid nodeName
        return nullptr;
    }
    return nullptr;
}
bool NodeManager::removeNode(const std::string& nodeName)
{

    return false;
}

bool NodeManager::removeNode(ObjectId oid)
{
    auto path = getNode(oid)->fullTreeName;
    // THIS ISN"T WORKING
    m_nodeTree.erase(path);
    delete m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(oid.m_ConstructorId)->GetConstructedObject(oid.m_PerTypeId);
    return true;
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
    Node* ptr = nullptr;
    try
    {
        ptr = m_nodeTree.get<Node*>(treeName);
    }catch(boost::exception &err)
    {
        std::cout << "Error getting node by name: " << treeName << std::endl;
    }

    return ptr;
}

void 
NodeManager::updateTreeName(Node* node, const std::string& prevTreeName)
{
	m_nodeTree.put(t_nodeTree::path_type{ node->fullTreeName, '.' }, node);
	m_nodeTree.erase(prevTreeName);
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
	auto nodePtr = getParent(sourceNode);
	if (!nodePtr)
		return;

	auto node = m_nodeTree.get_child(getParent(sourceNode)->fullTreeName);
	for (auto itr = node.begin(); itr != node.end(); ++itr)
	{
		output.push_back(itr->second.get_value<Node*>());
	}	
}

Node*
NodeManager::getParent(const std::string& sourceNode)
{
	auto idx = sourceNode.find_last_of('.');
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
    return node;
}
void NodeManager::getParentNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	Node* parent = getParent(sourceNode);
	if (!parent)
		return;
	output.push_back(parent);
	getParentNodes(parent->fullTreeName, output);
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
