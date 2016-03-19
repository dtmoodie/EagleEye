#pragma once
#include "../Defs.hpp"
#include <vector>
#include <string>
#include <map>
#include <IObject.h>
#include <opencv2/core/persistence.hpp>
#include <functional>
#include <parameters/Parameters.hpp>
#include "EagleLib/rcc/shared_ptr.hpp"

namespace EagleLib
{
    namespace Nodes
    {
        class Node;
    }
	

	class EAGLE_EXPORTS NodeManager
	{
	public:
		static NodeManager& getInstance();

		void RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo);

		std::vector<char const*> GetNodeInfo(std::string& nodeName);

		shared_ptr<Nodes::Node> addNode(const std::string& nodeName);

		std::vector<shared_ptr<Nodes::Node>> loadNodes(const std::string& saveFile);

		void saveNodes(std::vector<shared_ptr<Nodes::Node>>& topLevelNodes, const std::string& fileName);

		void saveNodes(std::vector<shared_ptr<Nodes::Node>>& topLevelNodes, cv::FileStorage fs);

		void printNodeTree(std::string* ret = nullptr);
		void saveTree(const std::string& fileName);
		std::string getNodeFile(const ObjectId& id);
		void onNodeRecompile(Nodes::Node* node);

        Nodes::Node* getNode(const ObjectId& id);
        Nodes::Node* getNode(const std::string& treeName);
		bool removeNode(const std::string& nodeName);
		bool removeNode(ObjectId oid);

		void updateTreeName(Nodes::Node* node, const std::string& prevTreeName);
		
		
		void getSiblingNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

		void getParentNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

		void getAccessibleNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

        Nodes::Node* getParent(const std::string& sourceNode);

		std::vector<std::string> getConstructableNodes();
        std::vector<std::string> getParametersOfType(std::function<bool(Loki::TypeInfo)> selector);
		virtual void OnConstructorsAdded();
	private:
		NodeManager();
		virtual ~NodeManager();
		std::vector<weak_ptr<Nodes::Node>>                         nodes;
		std::map<std::string, std::vector<char const*>>		m_nodeInfoMap;
	}; // class NodeManager
}
