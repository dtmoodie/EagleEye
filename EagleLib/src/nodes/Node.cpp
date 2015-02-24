#include "nodes/Node.h"
#include <opencv2/highgui.hpp>
#include <regex>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>

using namespace EagleLib;
#ifdef RCC_ENABLED
#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "../RuntimeObjectSystem/ISimpleSerializer.h"
#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cuda")
#endif


#endif

//static std::map<std::string, NodeFactory*> NodeFactories = std::map<std::string, NodeFactory*>();
boost::shared_ptr<Node>
Node::create(std::string& name)
{
    //if((*NodeFactories)[name])
    //	return (*NodeFactories)[name]->create();
    //return boost::shared_ptr<Node>();
	return Node::Ptr();
}
boost::shared_ptr<Node>
Node::create(const std::string& name)
{
    /*if (NodeFactories)
    {
		if ((*NodeFactories)[name])
			return (*NodeFactories)[name]->create();
	}else
	{
		std::cout << "Node factories is NULL, nothing has been registered yet" << std::endl;
	}
    return boost::shared_ptr<Node>();*/
	return Node::Ptr();
}
/*
void // This gets called before NodeFactories gets initialized :/ and so it breaks.
Node::registerType(const std::string& name, NodeFactory* factory)
{
	if (NodeFactories == NULL)
	{
		NodeFactories = new std::map<std::string, NodeFactory*>();
	}
	(*NodeFactories)[name] = factory;
}
*/
Node::Node()
{
	treeName = nodeName;
    parent = nullptr;
    enabled = true;
}

Node::~Node()
{

}
void
Node::getInputs()
{

}
Node::Ptr Node::addChild(Node* child)
{
    return addChild(boost::shared_ptr<Node>(child));
}
Node::Ptr Node::addChild(const boost::shared_ptr<Node>& child)
{
	if (!child)
		return child;
    if(errorCallback)
        child->errorCallback = errorCallback;
    if(statusCallback)
        child->statusCallback = statusCallback;
    if(warningCallback)
        child->warningCallback = warningCallback;
    for(int i = 0; i < child->parameters.size(); ++i)
        childParameters.push_back(std::make_pair(i,child->parameters[i]));
    int count = children.get<TreeName>().count(child->nodeName);

    child->treeName = child->treeName + "-" + boost::lexical_cast<std::string>(count);
	child->fullTreeName = this->treeName + "/" + child->treeName;
    child->parent = this;
    children.get<0>().push_back(child);
	// Check if this name already exists, if it does, increment 
    //children.push_back(child);
    //children[child->treeName] = child;
    return child;
}


boost::shared_ptr<Node>
Node::getChild(const std::string& name)
{
    auto it = children.get<TreeName>().find(name);
    if(it != children.get<TreeName>().end())
        return *it;
    return Node::Ptr();
}

boost::shared_ptr<Node>
Node::getChild(std::string name)
{
    auto it = children.get<TreeName>().find(name);
    if(it != children.get<TreeName>().end())
        return *it;
    return Node::Ptr();
}

boost::shared_ptr<Node>
Node::getChildRecursive(std::string treeName_)
{
    boost::shared_ptr<Node> ptr;
    // TODO tree structure parsing and correct directing of the search
    // Find the common base between this node and treeName


    return ptr;
}

void
Node::removeChild(boost::shared_ptr<Node> child)
{
    auto itr = children.get<NodeName>().find(child->treeName);
    if(itr != children.get<NodeName>().end())
        children.get<NodeName>().erase(itr);
}

void
Node::removeChild(const std::string &name)
{
    auto itr = children.get<NodeName>().find(name);
    if(itr != children.get<NodeName>().end())
        children.get<NodeName>().erase(itr);

}

cv::cuda::GpuMat
Node::process(cv::cuda::GpuMat &img)
{
    try
    {
        if(enabled)
            return doProcess(img);
    }catch(cv::Exception &e)
    {
        if(errorCallback)
        {
         std::string message = nodeName + std::string(e.what());
            errorCallback(message);
        }
    }catch(std::exception &e)
    {
        if(errorCallback)
        {
         std::string message = nodeName + std::string(e.what());
            errorCallback(message);
        }
    }
    return img;
}
void					
Node::process(cv::InputArray in, cv::OutputArray out)
{
	try
	{
		return doProcess(in, out);
	}
	catch (cv::Exception &e)
	{
		if (errorCallback)
		{
            std::string message = nodeName + std::string(e.what());
			errorCallback(message);
		}
	}
}

cv::cuda::GpuMat
Node::doProcess(cv::cuda::GpuMat& img)
{
    return img;
}
void					
Node::doProcess(cv::InputArray, cv::OutputArray)
{

}

void
Node::doProcess(cv::cuda::GpuMat& img, boost::promise<cv::cuda::GpuMat> &retVal)
{
    retVal.set_value(process(img));
}
void
Node::doProcess(cv::InputArray in, boost::promise<cv::OutputArray> &retVal)
{
	// Figure this out later :(

	
}
void
Node::registerDisplayCallback(boost::function<void(cv::Mat)>& f)
{
    cpuDisplayCallback = f;
}

void
Node::registerDisplayCallback(boost::function<void(cv::cuda::GpuMat)>& f)
{
	gpuDisplayCallback = f;
}

void
Node::spawnDisplay()
{
	cv::namedWindow(treeName);
	externalDisplay = true;
}
void
Node::killDisplay()
{
	if (externalDisplay)
		cv::destroyWindow(treeName);
}
std::string
Node::getName() const
{
    return nodeName;
}
std::string
Node::getTreeName() const
{
    return treeName;
}


Node::Ptr
Node::swap(const Node::Ptr &other)
{
    // By moving ownership of all parameters to the new node, all
    other->parameters = parameters;
    other->children = children;
    for (auto it = children.begin(); it != children.end(); ++it)
    {
        (*it)->parent = other.get();
    }
    return other;
}
void
Node::Init(bool firstInit)
{

}

void
Node::Init(const std::string &configFile)
{

}

void
Node::Init(const cv::FileNode& configNode)
{

}

void
Node::Serialize(ISimpleSerializer *pSerializer)
{
    IObject::Serialize(pSerializer);
    std::cout << "Serializing node" << std::endl;
    //SERIALIZE(children);
    SERIALIZE(parameters);
    SERIALIZE(children);
    SERIALIZE(treeName);
    SERIALIZE(nodeName);
    //SERIALIZEIOBJPTR(parent);
}
