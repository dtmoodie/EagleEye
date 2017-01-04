#include "EagleLib/IDataStream.hpp"
#include "EagleLib/Nodes/NodeFactory.h"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/NodeInfo.hpp"

#include <MetaObject/MetaObjectFactory.hpp>
#include <MetaObject/Parameters/InputParameter.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include "AUArray.h"
using namespace EagleLib;

NodeFactory* NodeFactory::Instance()
{
    static NodeFactory instance;
    return &instance;
}

NodeFactory::NodeFactory()
{
}

NodeFactory::~NodeFactory()
{

}


void NodeFactory::onConstructorsAdded()
{
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(IID_NodeObject);
    std::vector<Nodes::Node*> newNodes;
    for (size_t i = 0; i < constructors.size(); ++i)
    {
        size_t numObjects = constructors[i]->GetNumberConstructedObjects();
        for (size_t j = 0; j < numObjects; ++j)
        {
            auto ptr = constructors[i]->GetConstructedObject(j);
            if (ptr)
            {
                ptr = ptr->GetInterface(IID_NodeObject);
                if (ptr)
                {
                    auto nodePtr = static_cast<Nodes::Node*>(ptr);
                    newNodes.push_back(nodePtr);
                }
            }
        }
    }
    for (size_t i = 0; i < newNodes.size(); ++i)
    {
        auto parameters = newNodes[i]->GetDisplayParameters();
        for (size_t j = 0; j < parameters.size(); ++j)
        {
            if (parameters[j]->CheckFlags(mo::Input_e))
            {
                auto inputParam = dynamic_cast<mo::InputParameter*>(parameters[j]);
                if(inputParam)
                    inputParam->SetInput();
            }
        }
    }
}

rcc::shared_ptr<Nodes::Node> NodeFactory::AddNode(const std::string &nodeName)
{
    auto pConstructor = mo::MetaObjectFactory::Instance()->GetConstructor(nodeName.c_str());

    if (pConstructor && pConstructor->GetInterfaceId() == Nodes::Node::s_interfaceID)
    {
        IObject* pObj = pConstructor->Construct();
        IObject* interface = pObj->GetInterface(Nodes::Node::s_interfaceID);

        if (interface)
        {
            Nodes::Node* node = static_cast<Nodes::Node*>(interface);
            try
            {
                node->Init(true);
            }
            catch (cv::Exception &e)
            {
                LOG(error) << "Failed to initialize node " << nodeName << " due to: " << e.what();
                return rcc::shared_ptr<Nodes::Node>();
            }
            catch (...)
            {
                LOG(error) << "Failed to initialize node " << nodeName;
                return rcc::shared_ptr<Nodes::Node>();
            }

            nodes.push_back(rcc::weak_ptr<Nodes::Node>(node));
            return Nodes::Node::Ptr(node);
        }
        else
        {
            LOG(warning) << "[ NodeManager ] " << nodeName << " not a node";
            // Input nodename is a compatible object but it is not a node
            return rcc::shared_ptr<Nodes::Node>();
        }
    }
    else
    {
        LOG(warning) << "[ NodeManager ] " << nodeName << " not a valid node name";
        return rcc::shared_ptr<Nodes::Node>();
    }

    return rcc::shared_ptr<Nodes::Node>();
}
// WIP needs to be tested for complex dependency trees
std::vector<rcc::shared_ptr<Nodes::Node>> NodeFactory::AddNode(const std::string& nodeName, IDataStream* parentStream)
{
    IObjectConstructor* pConstructor = mo::MetaObjectFactory::Instance()->GetConstructor(nodeName.c_str());
    std::vector<rcc::shared_ptr<Nodes::Node>> constructed_nodes;
    if (pConstructor && pConstructor->GetInterfaceId() == Nodes::Node::s_interfaceID)
    {
        auto obj_info = pConstructor->GetObjectInfo();
        auto node_info = dynamic_cast<Nodes::NodeInfo*>(obj_info);
        auto parental_deps = node_info->GetParentalDependencies();
        // Since a data stream is selected and by definition a parental dependency must be in the direct parental path,
        // we build all parent dependencies
        rcc::shared_ptr<Nodes::Node> parent_node;
        for (auto& parent_dep : parental_deps)
        {
            if (parent_dep.size())
            {
                if (parent_node)
                {
                    auto parent_nodes = AddNode(parent_dep[0], parent_node.Get());
                    constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                    parent_node = parent_nodes.back();
                }
                else
                {
                    auto parent_nodes = AddNode(parent_dep[0], parentStream);
                    constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                    parent_node = parent_nodes.back();
                }
            }
        }
        auto non_parent_deps = node_info->GetNonParentalDependencies();
        auto existing_nodes = parentStream->GetNodes();
        for (auto & non_parent_dep : non_parent_deps)
        {
            bool found = false;
            for (auto& existing_node : existing_nodes)
            {
                for (auto& dep : non_parent_dep)
                {
                    if (existing_node->GetTypeName() == dep)
                    {
                        found = true;
                        break;
                    }
                }
            }
            // No qualified parental dependency was found, add first best candidate
            if (!found)
            {
                auto added_nodes = AddNode(non_parent_dep[0], parentStream);
                constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
            }
        }
        auto dependent_variable_nodes = node_info->CheckDependentVariables(parentStream->GetVariableManager().get());
        for (auto& dependent_variable_node : dependent_variable_nodes)
        {
            auto added_nodes = AddNode(dependent_variable_node, parentStream);
            constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
        }
        // All dependencies have been handled, construct node
        auto pNode = static_cast<EagleLib::Nodes::Node*>(pConstructor->Construct());
        pNode->Init(true);
        nodes.push_back(rcc::weak_ptr<Nodes::Node>(pNode));
        rcc::shared_ptr<Nodes::Node> node(pNode);
        constructed_nodes.push_back(node);
        if (parent_node)
        {
            parent_node->AddChild(node);
        }
        else
        {
            parentStream->AddNode(node);
        }
    }
    return constructed_nodes;
}

// recursively checks if a node exists in the parent hierarchy
bool check_parent_exists(Nodes::Node* node, const std::string& name)
{
    if (node->GetTypeName() == name)
        return true;
    /*if (auto parent = node->getParent())
        return check_parent_exists(parent, name);*/
    return false;
}

std::vector<rcc::shared_ptr<Nodes::Node>> NodeFactory::AddNode(const std::string& nodeName, Nodes::Node* parentNode)
{
    IObjectConstructor* pConstructor = mo::MetaObjectFactory::Instance()->GetConstructor(nodeName.c_str());
    std::vector<rcc::shared_ptr<Nodes::Node>> constructed_nodes;
    if (pConstructor && pConstructor->GetInterfaceId() == Nodes::Node::s_interfaceID)
    {
        auto obj_info = pConstructor->GetObjectInfo();
        auto node_info = dynamic_cast<Nodes::NodeInfo*>(obj_info);
        auto parental_deps = node_info->GetParentalDependencies();
        rcc::shared_ptr<Nodes::Node> parent_node;
        for (auto& parent_dep : parental_deps)
        {
            if (parent_dep.size())
            {
                // For each node already in this tree, search for any of the allowed parental dependencies
                bool found = false;
                for (auto& dep : parent_dep)
                {
                    if (check_parent_exists(parentNode, dep))
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    if (parent_node)
                    {
                        auto parent_nodes = AddNode(parent_dep[0], parent_node.Get());
                        constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                    }
                    else
                    {
                        auto parent_nodes = AddNode(parent_dep[0], parentNode);
                        constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                        parent_node = parent_nodes.back();
                    }
                }
            }
        }
        auto non_parent_deps = node_info->GetNonParentalDependencies();
        auto existing_nodes = parentNode->GetDataStream()->GetNodes();
        for (auto & non_parent_dep : non_parent_deps)
        {
            bool found = false;
            for (auto& existing_node : existing_nodes)
            {
                for (auto& dep : non_parent_dep)
                {
                    if (existing_node->GetTypeName() == dep)
                    {
                        found = true;
                        break;
                    }
                }
            }
            // No qualified parental dependency was found, add first best candidate
            if (!found)
            {
                auto added_nodes = AddNode(non_parent_dep[0], parentNode);
                constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
            }
        }
        auto dependent_variable_nodes = node_info->CheckDependentVariables(parentNode->GetDataStream()->GetVariableManager().get());
        for (auto& dependent_variable_node : dependent_variable_nodes)
        {
            auto added_nodes = AddNode(dependent_variable_node, parentNode);
            constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
        }

        rcc::shared_ptr<Nodes::Node> node(pConstructor->Construct());
        node->Init(true);
        parentNode->AddChild(node);
        constructed_nodes.push_back(node);
    }
    return constructed_nodes;    
}



bool NodeFactory::RemoveNode(const std::string& nodeName)
{    
    return false;
}

std::string NodeFactory::GetNodeFile(const ObjectId& id)
{
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors();
    if (constructors.size() > id.m_ConstructorId)
    {
        return std::string(constructors[id.m_ConstructorId]->GetFileName());
    }
    return std::string();
}

bool NodeFactory::RemoveNode(ObjectId oid)
{
    return false;
}

void NodeFactory::RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo)
{
    m_nodeInfoMap[nodeName] = nodeInfo;
}

Nodes::NodeInfo* NodeFactory::GetNodeInfo(std::string& nodeName)
{
    auto constructor = mo::MetaObjectFactory::Instance()->GetConstructor(nodeName.c_str());
    if (constructor)
    {
        auto obj_info = constructor->GetObjectInfo();
        if (obj_info)
        {
            if (obj_info->GetInterfaceId() == Nodes::Node::s_interfaceID)
            {
                auto node_info = dynamic_cast<EagleLib::Nodes::NodeInfo*>(obj_info);
                if (node_info)
                {
                    return node_info;
                }
                
            }
        }
    }
    return nullptr;
}

void NodeFactory::SaveTree(const std::string &fileName)
{
    
}

void
NodeFactory::onNodeRecompile(Nodes::Node *node)
{
}

Nodes::Node* NodeFactory::GetNode(const ObjectId& id)
{
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors();
    if (!id.IsValid())
        return nullptr;
    if (id.m_ConstructorId >= constructors.size())
        return nullptr;
    if (id.m_PerTypeId >= constructors[id.m_ConstructorId]->GetNumberConstructedObjects())
        return nullptr;
    IObject* pObj = constructors[id.m_ConstructorId]->GetConstructedObject(id.m_PerTypeId);
    if (!pObj)
        return nullptr;
    pObj = pObj->GetInterface(Nodes::Node::s_interfaceID);
    if (!pObj)
        return nullptr;
    return static_cast<Nodes::Node*>(pObj);
}

Nodes::Node* NodeFactory::GetNode(const std::string &treeName)
{
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (nodes[i] != nullptr)
        {
            if (nodes[i]->GetTreeName() == treeName)
            {
                return nodes[i].Get();
            }
        }
    }
    return nullptr;
}

void NodeFactory::UpdateTreeName(Nodes::Node* node, const std::string& prevTreeName)
{
    
    
}


void NodeFactory::GetSiblingNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output)
{
    
}

void NodeFactory::printTreeHelper(std::stringstream& tree, int level, Nodes::Node* node)
{
    
    for (int i = 0; i < level; ++i)
    {
        tree << "+";
    }
    tree << node->GetTreeName() << std::endl;
    auto children = node->GetChildren();
    for (size_t i = 0; i < children.size(); ++i)
    {
        printTreeHelper(tree, level + 1, children[i].Get());
    }
}

void NodeFactory::PrintNodeTree(std::string* ret)
{
    std::stringstream tree;
    std::vector<rcc::weak_ptr<Nodes::Node>> parentNodes;
    // First get the top level nodes for the tree
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (nodes[i] != nullptr)
        {
            auto parent_nodes = nodes[i]->GetParents();
            parentNodes.insert(parentNodes.begin(), parent_nodes.begin(), parent_nodes.end());
        }
    }
    for (size_t i = 0; i < parentNodes.size(); ++i)
    {
        printTreeHelper(tree, 0, parentNodes[i].Get());
    }
    if (ret)
    {
        *ret = tree.str();
    }
    else
    {
        std::cout << tree.str() << std::endl;
    }
}

Nodes::Node* NodeFactory::GetParent(const std::string& sourceNode)
{
    
    return nullptr;
}
void NodeFactory::GetParentNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output)
{
    
}

void NodeFactory::GetAccessibleNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output)
{
    
    GetSiblingNodes(sourceNode, output);
    GetParentNodes(sourceNode, output);
}

std::vector<std::string> NodeFactory::GetConstructableNodes()
{
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors();
    std::vector<std::string> output;
    for (size_t i = 0; i < constructors.size(); ++i)
    {
        if (constructors[i])
        {
            if (constructors[i]->GetInterfaceId() == Nodes::Node::s_interfaceID)
                output.push_back(constructors[i]->GetName());
        }
        else
        {
            std::cout << "Null constructor idx " << i << std::endl;
        }
    }
    return output;
}

std::vector<std::string> NodeFactory::GetParametersOfType(std::function<bool(mo::TypeInfo)> selector)
{
    
    std::vector<std::string> parameters;
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        auto node_params = nodes[i]->GetParameters();
        for (size_t j = 0; j < node_params.size(); ++j)
        {
            if (selector(node_params[j]->GetTypeInfo()))
                parameters.push_back(node_params[j]->GetTreeName());
        }
    }
    return parameters;
}
