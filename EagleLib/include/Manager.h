#pragma once
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>
#include "../RuntimeObjectSystem/ObjectInterface.h"
#include "../RuntimeObjectSystem/IObjectFactorySystem.h"
#include "nodes/Node.h"


namespace EagleLib
{
class Node;
    class NodeManager: public IObjectFactoryListener
    {
    public:
        NodeManager();
        virtual ~NodeManager();

        boost::shared_ptr<Node> addNode(const std::string& nodeName);



        bool Init();

        bool MainLoop();

        virtual void OnConstructorsAdded();

        virtual bool CheckRecompile();

    private:
        boost::shared_ptr<IRuntimeObjectSystem> m_pRuntimeObjectSystem;
        boost::shared_ptr<ICompilerLogger> m_pCompileLogger;

        std::map<std::string, boost::shared_ptr<Node>>        m_children;


    };
};
