#pragma once
#include "EagleLib.h"

#include <boost/shared_ptr.hpp>
#include <map>
#include <string>
#include "../RuntimeObjectSystem/ObjectInterface.h"
#include "../RuntimeObjectSystem/IObjectFactorySystem.h"
#include <opencv2/core/cvdef.h>
#include <list>
#define WIN32_FW_USE_FINDFIRST_API 1

#define ADD_CONSTRUCTORS(managerObj)  \
	auto moduleInterface = PerModuleInterface::GetInstance();	\
	auto vecConstructors = moduleInterface->GetConstructors();	\
	AUDynArray<IObjectConstructor*> dynConstructors;			\
	dynConstructors.Resize(vecConstructors.size());				\
	for (int i = 0; i < vecConstructors.size(); ++i)			\
		dynConstructors[i] = vecConstructors[i];				\
	(managerObj).addConstructors(dynConstructors);


namespace EagleLib
{
    class Node;
    class NodeTreeLeaf
    {
        Node*               node;
        Node*               parent;
        std::list<Node*>    children;
    };

    class NodeTree
    {

    };

    class CV_EXPORTS NodeManager : public IObjectFactoryListener
    {
    public:
        NodeManager();
        virtual ~NodeManager();

        Node *addNode(const std::string& nodeName);

		void addConstructors(IAUDynArray<IObjectConstructor*> & constructors);

        bool Init();

        bool MainLoop();

        virtual void OnConstructorsAdded();

        virtual bool CheckRecompile();

        void onNodeRecompile(Node* node);
        Node* getNode(const ObjectId& id);
        Node* getNode(const std::string& treeName);

    private:


        boost::shared_ptr<IRuntimeObjectSystem>             m_pRuntimeObjectSystem;
        boost::shared_ptr<ICompilerLogger>                  m_pCompileLogger;
        std::map<std::string, std::vector<Node*> >          m_nodeMap;


        std::vector<boost::shared_ptr<Node> >               nodeHistory;
        std::list<Node*>                                    deletedNodes;
        std::list<ObjectId>                                 deletedNodeIDs;

    };
};
