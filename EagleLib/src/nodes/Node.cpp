#include "nodes/Node.h"
#include <opencv2/highgui.hpp>
#include <regex>
using namespace EagleLib;
#ifdef RCC_ENABLED
#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
REGISTERCLASS(Node);
#endif

std::map<std::string, NodeFactory*> Node::NodeFactories = std::map<std::string, NodeFactory*>();

boost::shared_ptr<Node>
Node::create(const std::string& name)
{
    if(NodeFactories[name])
        return NodeFactories[name]->create();
    return boost::shared_ptr<Node>();
}
void
Node::registerType(const std::string& name, NodeFactory* factory)
{
    NodeFactories[name] = factory;
    //NodeFactories.insert()
}

Node::Node()
{
	treeName = nodeName;
}

Node::~Node()
{

}
void
Node::getInputs()
{

}
int Node::addChild(Node* child)
{
    boost::shared_ptr<Node> ptr(child);
    return addChild(ptr);
}

int Node::addChild(boost::shared_ptr<Node> child)
{
    if(errorCallback)
        child->errorCallback = errorCallback;
    if(statusCallback)
        child->statusCallback = statusCallback;
    if(warningCallback)
        child->warningCallback = warningCallback;
    for(int i = 0; i < child->parameters.size(); ++i)
    {
        childParameters.push_back(std::make_pair(i,child->parameters[i]));
    }
	child->treeName = this->treeName + "/" + child->treeName + "-0";
	// Check if this name already exists, if it does, increment 
    children.push_back(child);
    return children.size() -1;
}


boost::shared_ptr<Node>
Node::getChild(int index)
{
    return children[index];
}

boost::shared_ptr<Node>
Node::getChild(std::string name)
{
    for(int i = 0; i < children.size(); ++i)
        if(children[i]->nodeName == name)
            return children[i];
    return boost::shared_ptr<Node>();
}

boost::shared_ptr<Node>
Node::getChildRecursive(std::string treeName_)
{
    boost::shared_ptr<Node> ptr;
    // TODO tree structure parsing and correct directing of the search
    // Find the common base between this node and treeName

    for(int i = 0; i < children.size(); ++i)
    {
        ptr = children[i]->getChildRecursive(treeName_);
        if(ptr)
            return ptr;
    }
    return ptr;
}

void
Node::removeChild(boost::shared_ptr<Node> child)
{
    for(int i = 0; i < children.size(); ++i)
    {
        if(child == children[i])
        {
            children.erase(children.begin() + i);
            for(int j = 0; j < childParameters.size(); ++j)
            {
                if(childParameters[j].first == i)
                    childParameters.erase(childParameters.begin() + j);
            }
        }
    }
}

void
Node::removeChild(int idx)
{
    children.erase(children.begin() + idx);
    for(int j = 0; j < childParameters.size(); ++j)
    {
        if(childParameters[j].first == idx)
            childParameters.erase(childParameters.begin() + j);
    }

}

cv::cuda::GpuMat
Node::process(cv::cuda::GpuMat &img)
{
    try
    {
        return doProcess(img);
    }catch(cv::Exception &e)
    {
        if(errorCallback)
        {
         std::string message = std::string(__FUNCTION__) + std::string(e.what());
            errorCallback(message);
        }
    }
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
			std::string message = std::string(__FUNCTION__) + std::string(e.what());
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
Node::getName()
{
    std::string name;
    if(parent != NULL)
        parent->getName() + "/" + name;
    return name;
}
