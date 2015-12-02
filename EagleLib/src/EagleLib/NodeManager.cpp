#include "NodeManager.h"
#include "ObjectManager.h"
#include "Nodes/Node.h"
#include <boost/filesystem.hpp>

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
	LOG_TRACE;
	AUDynArray<IObjectConstructor*> constructors;
	ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	std::vector<Node*> newNodes;
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
					auto nodePtr = static_cast<Node*>(ptr);
					newNodes.push_back(nodePtr);
				}
			}
		}
	}
	for (size_t i = 0; i < newNodes.size(); ++i)
	{
		for (size_t j = 0; j < newNodes[i]->parameters.size(); ++j)
		{
			if (newNodes[i]->parameters[j]->type & Parameters::Parameter::Input)
			{
				auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(newNodes[i]->parameters[j]);
				inputParam->SetInput(std::string());
			}
		}
	}
}

shared_ptr<Node> NodeManager::addNode(const std::string &nodeName)
{
	LOG_TRACE << nodeName;
	IObjectConstructor* pConstructor = ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(nodeName.c_str());

	if (pConstructor && pConstructor->GetInterfaceId() == IID_NodeObject)
	{
		IObject* pObj = pConstructor->Construct();
		IObject* interface = pObj->GetInterface(IID_NodeObject);

		if (interface)
		{
			Node* node = static_cast<Node*>(interface);
			try
			{
				node->Init(true);
			}
			catch (cv::Exception &e)
			{
				BOOST_LOG_TRIVIAL(error) << "Failed to initialize node " << nodeName << " due to: " << e.what();
				return shared_ptr<Node>();
			}
			catch (...)
			{
				BOOST_LOG_TRIVIAL(error) << "Failed to initialize node " << nodeName;
				return shared_ptr<Node>();
			}

			nodes.push_back(weak_ptr<Node>(node));
			return Node::Ptr(node);
		}
		else
		{
			BOOST_LOG_TRIVIAL(warning) << "[ NodeManager ] " << nodeName << " not a node";
			// Input nodename is a compatible object but it is not a node
			return shared_ptr<Node>();
		}
	}
	else
	{
		BOOST_LOG_TRIVIAL(warning) << "[ NodeManager ] " << nodeName << " not a valid node name";
		return shared_ptr<Node>();
	}

	return shared_ptr<Node>();
}
std::vector<shared_ptr<Node>> NodeManager::loadNodes(const std::string& saveFile)
{
	LOG_TRACE;
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
	std::vector<shared_ptr<Node>> nodes;
	nodes.reserve(nodeCount);
	for (int i = 0; i < nodeCount; ++i)
	{
		auto nodeFS = fs["Node-" + boost::lexical_cast<std::string>(i)];
		std::string name = (std::string)nodeFS["NodeName"];
		Node::Ptr node = addNode(name);
		node->Init(nodeFS);
		nodes.push_back(node);
	}
	return nodes;
}

void NodeManager::saveNodes(std::vector<shared_ptr<Node>>& topLevelNodes, const std::string& fileName)
{
	cv::FileStorage fs;
	fs.open(fileName, cv::FileStorage::WRITE);
	saveNodes(topLevelNodes, fs);
	fs.release();
}
void NodeManager::saveNodes(std::vector<shared_ptr<Node>>& topLevelNodes, cv::FileStorage fs)
{
	LOG_TRACE;
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
	LOG_TRACE;
	return false;
}
std::string NodeManager::getNodeFile(const ObjectId& id)
{
	LOG_TRACE;
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
	LOG_TRACE;
	return false;
}
void NodeManager::RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo)
{
	m_nodeInfoMap[nodeName] = nodeInfo;
}
std::vector<const char*>& NodeManager::GetNodeInfo(std::string& nodeName)
{
	return m_nodeInfoMap[nodeName];
}






void NodeManager::saveTree(const std::string &fileName)
{
	LOG_TRACE;
}

void
NodeManager::onNodeRecompile(Node *node)
{
}

Node*
NodeManager::getNode(const ObjectId& id)
{
	LOG_TRACE;
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
	return static_cast<Node*>(pObj);
}

Node*
NodeManager::getNode(const std::string &treeName)
{
	LOG_TRACE;
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		if (nodes[i] != nullptr)
		{
			if (nodes[i]->fullTreeName == treeName)
			{
				return nodes[i].get();
			}
		}
	}
	return nullptr;
}

void
NodeManager::updateTreeName(Node* node, const std::string& prevTreeName)
{
	LOG_TRACE;
	/*m_nodeTree.put(t_nodeTree::path_type{ node->fullTreeName, '.' }, node);
	m_nodeTree.erase(prevTreeName);*/
}

void
NodeManager::addParameters(Node* node)
{
	LOG_TRACE;
	for (size_t i = 0; i < node->parameters.size(); ++i)
	{

	}
}

Parameters::Parameter::Ptr
NodeManager::getParameter(const std::string& name)
{
	LOG_TRACE;
	// Strip off the path for the node
	auto idx = name.find(':');
	std::string parameterName = name.substr(idx + 1);
	auto node = getNode(name.substr(0, idx));
	if (node == nullptr)
		return Parameters::Parameter::Ptr();
	return node->getParameter(parameterName);
}

void
NodeManager::getSiblingNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	LOG_TRACE;
}

void printTreeHelper(std::stringstream& tree, int level, Node* node)
{
	LOG_TRACE;
	for (int i = 0; i < level; ++i)
	{
		tree << "+";
	}
	tree << node->fullTreeName << std::endl;
	for (size_t i = 0; i < node->children.size(); ++i)
	{
		printTreeHelper(tree, level + 1, node->children[i].get());
	}
}

void NodeManager::printNodeTree(std::string* ret)
{
	LOG_TRACE;
	std::stringstream tree;
	std::vector<weak_ptr<Node>> parentNodes;
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

Node*
NodeManager::getParent(const std::string& sourceNode)
{
	LOG_TRACE;
	return nullptr;
}
void NodeManager::getParentNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	LOG_TRACE;
}

void NodeManager::getAccessibleNodes(const std::string& sourceNode, std::vector<Node*>& output)
{
	LOG_TRACE;
	getSiblingNodes(sourceNode, output);
	getParentNodes(sourceNode, output);
}
std::vector<std::string>
NodeManager::getConstructableNodes()
{
	LOG_TRACE;
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

std::vector<std::string> NodeManager::getParametersOfType(boost::function<bool(Loki::TypeInfo)> selector)
{
	LOG_TRACE;
	std::vector<std::string> parameters;
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		for (size_t j = 0; j < nodes[i]->parameters.size(); ++j)
		{
			if (selector(nodes[i]->parameters[j]->GetTypeInfo()))
				parameters.push_back(nodes[i]->parameters[j]->GetTreeName());
		}
	}
	return parameters;
}