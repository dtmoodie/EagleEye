#pragma once
#include "EagleLib/Detail/Export.hpp"
#include <shared_ptr.hpp>
#include <IObject.h>


#include <opencv2/core/persistence.hpp>

#include <functional>
#include <vector>
#include <string>
#include <map>

namespace EagleLib
{
    namespace Nodes
    {
        class Node;
        struct NodeInfo;
    }
    class IDataStream;

    class EAGLE_EXPORTS NodeFactory
    {
    public:
        static NodeFactory* Instance();

        void RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo);

        Nodes::NodeInfo* GetNodeInfo(std::string& nodeName);

        rcc::shared_ptr<Nodes::Node> AddNode(const std::string& nodeName);

        // Adds a node by name to the data stream or the parent node.  
        std::vector<rcc::shared_ptr<Nodes::Node>> AddNode(const std::string& nodeName, IDataStream* parentStream);
        std::vector<rcc::shared_ptr<Nodes::Node>> AddNode(const std::string& nodeName, Nodes::Node* parentNode);


        

        

        void PrintNodeTree(std::string* ret = nullptr);
        void SaveTree(const std::string& fileName);
        std::string GetNodeFile(const ObjectId& id);

        Nodes::Node* GetNode(const ObjectId& id);
        Nodes::Node* GetNode(const std::string& treeName);
        bool RemoveNode(const std::string& nodeName);
        bool RemoveNode(ObjectId oid);

        void UpdateTreeName(Nodes::Node* node, const std::string& prevTreeName);
        
        
        void GetSiblingNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

        void GetParentNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

        void GetAccessibleNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

        Nodes::Node* GetParent(const std::string& sourceNode);

        std::vector<std::string> GetConstructableNodes();
        std::vector<std::string> GetParametersOfType(std::function<bool(mo::TypeInfo)> selector);

        
    private:
        void printTreeHelper(std::stringstream& tree, int level, Nodes::Node* node);
        void onNodeRecompile(Nodes::Node* node);
        virtual void onConstructorsAdded();
        NodeFactory();
        virtual ~NodeFactory();
        std::vector<rcc::weak_ptr<Nodes::Node>>                         nodes;
        std::map<std::string, std::vector<char const*>>        m_nodeInfoMap;
    }; // class NodeManager
}
