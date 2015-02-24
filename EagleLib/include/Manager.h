#pragma once
#include "nodes/Node.h"
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>
#include "../RuntimeObjectSystem/ObjectInterface.h"
#include "../RuntimeObjectSystem/IObjectFactorySystem.h"




namespace EagleLib
{
class Node;
    class NodeManager: public IObjectFactoryListener
    {
    public:
        NodeManager();
        virtual ~NodeManager();

        Node *addNode(const std::string& nodeName);

        bool Init();

        bool MainLoop();

        virtual void OnConstructorsAdded();

        virtual bool CheckRecompile();

    private:


        boost::shared_ptr<IRuntimeObjectSystem> m_pRuntimeObjectSystem;
        boost::shared_ptr<ICompilerLogger> m_pCompileLogger;
        //boost::shared_ptr<Node> node;
        //Node* node;
        std::map<std::string, std::vector<Node*> >        m_nodeMap;

        // RCC works by swapping the function pointers in already existing objects.
        // To prevent memory leaking
        std::vector<boost::shared_ptr<Node> > nodeHistory;

    };
};
