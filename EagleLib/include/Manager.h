#pragma once

#include "EagleLib.h"

#include <boost/shared_ptr.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <string>
#include "../RuntimeObjectSystem/ObjectInterface.h"
#include "../RuntimeObjectSystem/IObjectFactorySystem.h"
#include <opencv2/core/cvdef.h>
#include <list>



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
	class Parameter;

    class CV_EXPORTS NodeManager : public IObjectFactoryListener
    {

	public:
		static NodeManager& getInstance();

        Node *addNode(const std::string& nodeName);

		void addConstructors(IAUDynArray<IObjectConstructor*> & constructors);
		void setupModule(IPerModuleInterface* pPerModuleInterface);
        bool Init();

        bool MainLoop();

        virtual void OnConstructorsAdded();

        virtual bool CheckRecompile();

        void onNodeRecompile(Node* node);
        Node* getNode(const ObjectId& id);
        Node* getNode(const std::string& treeName);
		void updateTreeName(Node* node, const std::string& prevTreeName);
		void addParameters(Node* node);
		boost::shared_ptr< Parameter > getParameter(const std::string& name);

		// Returns a list of nodes ordered in closeness relative to sourceNode
		// If a tree is something like the following:
		//				A
		//			   /  \
		//			  B    C
		//          / | \    \
		//         D  E  F    G
		// Then calling getNodes("A.B.D", 1) will return a vector with [B,E,F]
		// Calling getNodes("A.B.D", 2) will return a vector with [B,E,F,A,C,G]
		// Calling getNodes("A.B", 0) will return a vector with [D,E,F] in insertion order

		void getSiblingNodes(const std::string& sourceNode, std::vector<Node*>& output);
		void getParentNodes(const std::string& sourceNode, std::vector<Node*>& output);
		void getAccessibleNodes(const std::string& sourceNode, std::vector<Node*>& output);
		Node* getParent(const std::string& sourceNode);

	private:
		NodeManager();
		virtual ~NodeManager();
        boost::shared_ptr<IRuntimeObjectSystem>             m_pRuntimeObjectSystem;
        boost::shared_ptr<ICompilerLogger>                  m_pCompileLogger;
        std::map<std::string, std::vector<Node*> >          m_nodeMap;


        std::vector<boost::shared_ptr<Node> >               nodeHistory;
        std::list<Node*>                                    deletedNodes;
        std::list<ObjectId>                                 deletedNodeIDs;
		typedef boost::property_tree::basic_ptree<std::string, Node*> t_nodeTree;
		//typedef boost::property_tree::ptree t_nodeTree;
		t_nodeTree m_nodeTree;
		
    };
};
