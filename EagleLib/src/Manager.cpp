#include "Manager.h"
#include <RuntimeObjectSystem.h>
#include "StdioLogSystem.h"
//#include <IObjectUtils.h>
using namespace EagleLib;

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


}

bool
NodeManager::MainLoop()
{

}

void
NodeManager::OnConstructorsAdded()
{
    // Recompiling has occured.  Testing just one node for now



}

boost::shared_ptr<Node> NodeManager::addNode(const std::string &nodeName)
{
    IObjectConstructor* pConstructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());

    if(pConstructor)
    {
        IObject* pObj = pConstructor->Construct();
        void* interface = pObj->GetInterface(IID_NodeObject);
        if(interface)
        {
            Node::Ptr ptr(static_cast<Node*>(interface));
            m_children[ptr->nodeName] = ptr;
            return ptr;
        }else
        {
            // Input nodename is a compatible object but it is not a node
            return Node::Ptr();
        }
    }else
    {
        // Invalid nodeName
        return Node::Ptr();
    }
    return Node::Ptr();
}

bool NodeManager::CheckRecompile()
{
    if( m_pRuntimeObjectSystem->GetIsCompiledComplete())
    {
        m_pRuntimeObjectSystem->LoadCompiledModule();
    }

    if(!m_pRuntimeObjectSystem->GetIsCompiling())
    {
        static int numItrs = 0;
        std::cout << "Iteration: " << numItrs++ << std::endl;
        m_pRuntimeObjectSystem->GetFileChangeNotifier()->Update(1.0f);
        usleep(1000*1000);
    }
}
