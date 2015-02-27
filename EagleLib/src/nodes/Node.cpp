#include "nodes/Node.h"
#include <opencv2/highgui.hpp>
#include <regex>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "Manager.h"
using namespace EagleLib;
#ifdef RCC_ENABLED
#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "../RuntimeObjectSystem/ISimpleSerializer.h"
#if __linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cuda")
#endif


#endif

Node::Node()
{
	treeName = nodeName;
    parent.SetInvalid();
    enabled = true;
    nodeManager = nullptr;

}

Node::~Node()
{
    if(nodeManager)
        nodeManager->onNodeRecompile(this);
}
void
Node::getInputs()
{

}
Node*
Node::addChild(Node* child)
{
    if (!child || !nodeManager)
        return child;
    if(errorCallback)
        child->errorCallback = errorCallback;
    if(statusCallback)
        child->statusCallback = statusCallback;
    if(warningCallback)
        child->warningCallback = warningCallback;

    int count = children.get<NodeName>().count(child->nodeName);

    child->treeName = child->treeName + "-" + boost::lexical_cast<std::string>(count);
	child->fullTreeName = this->treeName + "/" + child->treeName;
    child->parent = GetObjectId();
    NodeInfo info;
    info.id = child->GetObjectId();
    info.index = children.get<0>().size();
    info.nodeName = child->nodeName;
    info.treeName = child->treeName;
    children.get<0>().push_back(info);
    return child;
}


Node*
Node::getChild(const std::string& treeName)
{
    auto itr = children.get<TreeName>().find(treeName);
    if(itr == children.get<TreeName>().end())
        return nullptr;
    return nodeManager->getNode(itr->id);
}


Node*
Node::getChild(const int& index)
{
    auto itr = children.get<0>()[index];
    return nodeManager->getNode(itr.id);
}

Node*
Node::getChild(const ObjectId& id)
{
    return nodeManager->getNode(id);
}

Node*
Node::getChildRecursive(std::string treeName_)
{

    // TODO tree structure parsing and correct directing of the search
    // Find the common base between this node and treeName


    return nullptr;
}

void
Node::removeChild(ObjectId childId)
{


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


Node*
Node::swap(Node* other)
{
    // By moving ownership of all parameters to the new node, all

    return other;
}
void
Node::Init(bool firstInit)
{
    if(firstInit)
        m_OID = GetObjectId();
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



REGISTERCLASS(Node)
