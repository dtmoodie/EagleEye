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

	m_pRuntimeObjectSystem->AddLibraryDir(BOOST_LIB_DIR);
	m_pRuntimeObjectSystem->AddLibraryDir(OPENCV_LIB_DIR);

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
    // Recompiling has occured.  Testing just one node for now
    for(auto itr = m_nodeMap.begin(); itr != m_nodeMap.end(); ++itr)
    {
        IObjectConstructor* pConstructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(itr->first.c_str());
        if(pConstructor)
        {

            for(int i = 0; i < pConstructor->GetNumberConstructedObjects(); ++i)
            {
                IObject* pObj = pConstructor->GetConstructedObject(i);

                pObj = pObj->GetInterface(IID_NodeObject);
                if(pObj)
                {
                    Node* pNode = dynamic_cast<Node*>(pObj);
                    for(int j = 0; j < itr->second.size(); ++j)
                    {
                        if(itr->second[j]->treeName == pNode->treeName)
                        {
                            itr->second[j] = pNode;
                            // For some reason, trying to delete ptr yields a segfault.  Maybe the objectfactory is already handling this?
                            // Need to figure out how to notify the object factory of a delete in the case of a legitimate delete
                        }
                    }
                }
            }

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
            node->nodeManager = this;
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
void 
NodeManager::addConstructors(IAUDynArray<IObjectConstructor*> & constructors)
{
	m_pRuntimeObjectSystem->GetObjectFactorySystem()->AddConstructors(constructors);
}

bool NodeManager::CheckRecompile()
{
	static boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
	boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
	boost::posix_time::time_duration delta = currentTime - prevTime;
    if( m_pRuntimeObjectSystem->GetIsCompiledComplete())
    {
        m_pRuntimeObjectSystem->LoadCompiledModule();
    }

    if(!m_pRuntimeObjectSystem->GetIsCompiling())
    {
        m_pRuntimeObjectSystem->GetFileChangeNotifier()->Update(float(delta.total_milliseconds())/1000.0);
    }
	prevTime = currentTime;
	return true;
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
    return nullptr;
}
