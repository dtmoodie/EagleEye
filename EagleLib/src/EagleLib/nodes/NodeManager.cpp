#include "NodeManager.h"
#include "EagleLib/rcc/ObjectManager.h"
#include "EagleLib/DataStreamManager.h"
#include "Node.h"

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
using namespace EagleLib;

NodeManager& NodeManager::getInstance()
{
	static NodeManager instance;
	return instance;
}

NodeManager::NodeManager()
{
	ObjectManager::Instance().RegisterConstructorAddedCallback(boost::bind(&NodeManager::OnConstructorsAdded, this));
}

NodeManager::~NodeManager()
{

}


void
NodeManager::OnConstructorsAdded()
{
	
	AUDynArray<IObjectConstructor*> constructors;
	ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	std::vector<Nodes::Node*> newNodes;
	for (size_t i = 0; i < constructors.Size(); ++i)
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
        auto parameters = newNodes[i]->getParameters();
		for (size_t j = 0; j < parameters.size(); ++j)
		{
			if (parameters[j]->type & Parameters::Parameter::Input)
			{
				auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(parameters[j]);
				inputParam->SetInput(std::string());
			}
		}
	}
}

rcc::shared_ptr<Nodes::Node> NodeManager::addNode(const std::string &nodeName)
{
	IObjectConstructor* pConstructor = ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());

	if (pConstructor && pConstructor->GetInterfaceId() == IID_NodeObject)
	{
		IObject* pObj = pConstructor->Construct();
		IObject* interface = pObj->GetInterface(IID_NodeObject);

		if (interface)
		{
			Nodes::Node* node = static_cast<Nodes::Node*>(interface);
			try
			{
				//node->Init(true);
			}
			catch (cv::Exception &e)
			{
				BOOST_LOG_TRIVIAL(error) << "Failed to initialize node " << nodeName << " due to: " << e.what();
				return rcc::shared_ptr<Nodes::Node>();
			}
			catch (...)
			{
				BOOST_LOG_TRIVIAL(error) << "Failed to initialize node " << nodeName;
				return rcc::shared_ptr<Nodes::Node>();
			}

			nodes.push_back(rcc::weak_ptr<Nodes::Node>(node));
			return Nodes::Node::Ptr(node);
		}
		else
		{
			BOOST_LOG_TRIVIAL(warning) << "[ NodeManager ] " << nodeName << " not a node";
			// Input nodename is a compatible object but it is not a node
			return rcc::shared_ptr<Nodes::Node>();
		}
	}
	else
	{
		BOOST_LOG_TRIVIAL(warning) << "[ NodeManager ] " << nodeName << " not a valid node name";
		return rcc::shared_ptr<Nodes::Node>();
	}

	return rcc::shared_ptr<Nodes::Node>();
}
// WIP needs to be tested for complex dependency trees
std::vector<rcc::shared_ptr<Nodes::Node>> NodeManager::addNode(const std::string& nodeName, DataStream* parentStream)
{
	IObjectConstructor* pConstructor = ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());
	std::vector<rcc::shared_ptr<Nodes::Node>> constructed_nodes;
	if (pConstructor && pConstructor->GetInterfaceId() == IID_NodeObject)
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
					auto parent_nodes = addNode(parent_dep[0], parent_node.get());
					constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
				}
				else
				{
					auto parent_nodes = addNode(parent_dep[0], parentStream);
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
					if (existing_node->getName() == dep)
					{
						found = true;
						break;
					}
				}
			}
			// No qualified parental dependency was found, add first best candidate
			if (!found)
			{
				auto added_nodes = addNode(non_parent_dep[0], parentStream);
				constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
			}
		}
		auto dependent_variable_nodes = node_info->CheckDependentVariables(parentStream->GetVariableManager().get());
		for (auto& dependent_variable_node : dependent_variable_nodes)
		{
			auto added_nodes = addNode(dependent_variable_node, parentStream);
			constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
		}
		// All dependencies have been handled, construct node
		auto pNode = static_cast<EagleLib::Nodes::Node*>(pConstructor->Construct());
		nodes.push_back(rcc::weak_ptr<Nodes::Node>(pNode));
		rcc::shared_ptr<Nodes::Node> node(pNode);
		constructed_nodes.push_back(node);
		if (parent_node)
		{
			parent_node->addChild(node);
		}
		else
		{
			parentStream->AddNode(node);
		}
	}
	return constructed_nodes;
}

std::vector<rcc::shared_ptr<Nodes::Node>> NodeManager::addNode(const std::string& nodeName, Nodes::Node* parentNode)
{
	IObjectConstructor* pConstructor = ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());
	std::vector<rcc::shared_ptr<Nodes::Node>> constructed_nodes;
	if (pConstructor && pConstructor->GetInterfaceId() == IID_NodeObject)
	{
		auto obj_info = pConstructor->GetObjectInfo();
		auto node_info = static_cast<Nodes::NodeInfo*>(obj_info);
		auto parental_deps = node_info->GetParentalDependencies();
		rcc::shared_ptr<Nodes::Node> parent_node;
		for (auto& parent_dep : parental_deps)
		{
			if (parent_dep.size())
			{
				if (parent_node)
				{
					auto parent_nodes = addNode(parent_dep[0], parent_node.get());
					constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
				}
				else
				{
					auto parent_nodes = addNode(parent_dep[0], parentNode);
					constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
					parent_node = parent_nodes.back();
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
					if (existing_node->getName() == dep)
					{
						found = true;
						break;
					}
				}
			}
			// No qualified parental dependency was found, add first best candidate
			if (!found)
			{
				auto added_nodes = addNode(non_parent_dep[0], parentNode);
				constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
			}
		}
		auto dependent_variable_nodes = node_info->CheckDependentVariables(parentNode->GetVariableManager());
		for (auto& dependent_variable_node : dependent_variable_nodes)
		{
			auto added_nodes = addNode(dependent_variable_node, parentNode);
			constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
		}

		rcc::shared_ptr<Nodes::Node> node(pConstructor->Construct());
		parentNode->addChild(node);
		constructed_nodes.push_back(node);
	}
	return constructed_nodes;	
}

std::vector<rcc::shared_ptr<Nodes::Node>> NodeManager::loadNodes(const std::string& saveFile)
{
	
	boost::filesystem::path path(saveFile);
	if (!boost::filesystem::is_regular_file(path))
	{
		//std::cout << "Unable to load " << saveFile << " doesn't exist, or is not a regular file" << std::endl;
		LOG_TRIVIAL(warning) << "[ NodeManager ] " << saveFile << " doesn't exist or not a regular file";
	}
	cv::FileStorage fs;
	try
	{
		fs.open(saveFile, cv::FileStorage::READ);
	}
	catch (cv::Exception &e)
	{
		//std::cout << e.what() << std::endl;
		LOG_TRIVIAL(error) << "[ NodeManager ] " << e.what();
	}

	int nodeCount = (int)fs["TopLevelNodeCount"];
	LOG_TRIVIAL(info) << "[ NodeManager ] " << "Loading " << nodeCount << " nodes";
	std::vector<rcc::shared_ptr<Nodes::Node>> nodes;
	nodes.reserve(nodeCount);
	for (int i = 0; i < nodeCount; ++i)
	{
		auto nodeFS = fs["Node-" + boost::lexical_cast<std::string>(i)];
		std::string name = (std::string)nodeFS["NodeName"];
		Nodes::Node::Ptr node = addNode(name);
		node->Init(nodeFS);
		nodes.push_back(node);
	}
	return nodes;
}

void NodeManager::saveNodes(std::vector<rcc::shared_ptr<Nodes::Node>>& topLevelNodes, const std::string& fileName)
{
	cv::FileStorage fs;
	fs.open(fileName, cv::FileStorage::WRITE);
	saveNodes(topLevelNodes, fs);
	fs.release();
}
void NodeManager::saveNodes(std::vector<rcc::shared_ptr<Nodes::Node>>& topLevelNodes, cv::FileStorage fs)
{
	
	fs << "TopLevelNodeCount" << (int)topLevelNodes.size();

	for (size_t i = 0; i < topLevelNodes.size(); ++i)
	{
		fs << "Node-" + boost::lexical_cast<std::string>(i) << "{";
		topLevelNodes[i]->Serialize(fs);
		fs << "}";
	}
}

bool NodeManager::removeNode(const std::string& nodeName)
{
	
	return false;
}
std::string NodeManager::getNodeFile(const ObjectId& id)
{
	
	AUDynArray<IObjectConstructor*> constructors;
	ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	if (constructors.Size() > id.m_ConstructorId)
	{
		return std::string(constructors[id.m_ConstructorId]->GetFileName());
	}
	return std::string();
}

bool NodeManager::removeNode(ObjectId oid)
{
	
	return false;
}
void NodeManager::RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo)
{
	m_nodeInfoMap[nodeName] = nodeInfo;
}
std::vector<const char*> NodeManager::GetNodeInfo(std::string& nodeName)
{
    auto constructor = ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());
    if (constructor)
    {
        auto obj_info = constructor->GetObjectInfo();
        if (obj_info)
        {
            if (obj_info->GetObjectInfoType() == 1)
            {
                auto node_info = dynamic_cast<EagleLib::Nodes::NodeInfo*>(obj_info);
                if (node_info)
                {
                    return node_info->GetNodeHierarchy();
                }
                
            }
        }
    }
    return std::vector<const char*>();
	//return m_nodeInfoMap[nodeName];
}






void NodeManager::saveTree(const std::string &fileName)
{
	
}

void
NodeManager::onNodeRecompile(Nodes::Node *node)
{
}

Nodes::Node*
NodeManager::getNode(const ObjectId& id)
{
	
	AUDynArray<IObjectConstructor*> constructors;
	ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	if (!id.IsValid())
		return nullptr;
	if (id.m_ConstructorId >= constructors.Size())
		return nullptr;
	if (id.m_PerTypeId >= constructors[id.m_ConstructorId]->GetNumberConstructedObjects())
		return nullptr;
	IObject* pObj = constructors[id.m_ConstructorId]->GetConstructedObject(id.m_PerTypeId);
	if (!pObj)
		return nullptr;
	pObj = pObj->GetInterface(IID_NodeObject);
	if (!pObj)
		return nullptr;
	return static_cast<Nodes::Node*>(pObj);
}

Nodes::Node*
NodeManager::getNode(const std::string &treeName)
{
	
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		if (nodes[i] != nullptr)
		{
            if (nodes[i]->getFullTreeName() == treeName)
			{
				return nodes[i].get();
			}
		}
	}
	return nullptr;
}

void
NodeManager::updateTreeName(Nodes::Node* node, const std::string& prevTreeName)
{
	
	
}


void
NodeManager::getSiblingNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output)
{
	
}

void printTreeHelper(std::stringstream& tree, int level, Nodes::Node* node)
{
	
	for (int i = 0; i < level; ++i)
	{
		tree << "+";
	}
	tree << node->getFullTreeName() << std::endl;
	for (size_t i = 0; i < node->children.size(); ++i)
	{
		printTreeHelper(tree, level + 1, node->children[i].get());
	}
}

void NodeManager::printNodeTree(std::string* ret)
{
	
	std::stringstream tree;
	std::vector<rcc::weak_ptr<Nodes::Node>> parentNodes;
	// First get the top level nodes for the tree
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		if (nodes[i] != nullptr)
		{
			if (nodes[i]->parent == nullptr)
			{
				parentNodes.push_back(nodes[i]);
			}
		}
	}
	for (size_t i = 0; i < parentNodes.size(); ++i)
	{
		printTreeHelper(tree, 0, parentNodes[i].get());
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

Nodes::Node*
NodeManager::getParent(const std::string& sourceNode)
{
	
	return nullptr;
}
void NodeManager::getParentNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output)
{
	
}

void NodeManager::getAccessibleNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output)
{
	
	getSiblingNodes(sourceNode, output);
	getParentNodes(sourceNode, output);
}
std::vector<std::string>
NodeManager::getConstructableNodes()
{
	
	AUDynArray<IObjectConstructor*> constructors;
	ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	std::vector<std::string> output;
	for (size_t i = 0; i < constructors.Size(); ++i)
	{
		if (constructors[i])
		{
			if (constructors[i]->GetInterfaceId() == IID_NodeObject)
				output.push_back(constructors[i]->GetName());
		}
		else
		{
			std::cout << "Null constructor idx " << i << std::endl;
		}
	}
	return output;
}

std::vector<std::string> NodeManager::getParametersOfType(std::function<bool(Loki::TypeInfo)> selector)
{
	
	std::vector<std::string> parameters;
	for (size_t i = 0; i < nodes.size(); ++i)
	{
        auto node_params = nodes[i]->getParameters();
		for (size_t j = 0; j < node_params.size(); ++j)
		{
			if (selector(node_params[j]->GetTypeInfo()))
				parameters.push_back(node_params[j]->GetTreeName());
		}
	}
	return parameters;
}
