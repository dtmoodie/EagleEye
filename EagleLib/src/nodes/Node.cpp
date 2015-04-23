#include "nodes/Node.h"
#include <opencv2/highgui.hpp>
#include <regex>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "Manager.h"
#include <boost/date_time.hpp>
using namespace EagleLib;
#ifdef RCC_ENABLED
#include "../RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "../RuntimeObjectSystem/ISimpleSerializer.h"

#if _WIN32
	#if _DEBUG
		RUNTIME_COMPILER_LINKLIBRARY("opencv_core300d.lib")
		RUNTIME_COMPILER_LINKLIBRARY("opencv_cuda300d.lib")
	#else
		RUNTIME_COMPILER_LINKLIBRARY("opencv_core300.lib")
		RUNTIME_COMPILER_LINKLIBRARY("opencv_cuda300.lib")
	#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core -lopencv_cuda")
#endif



#endif
Verbosity Node::debug_verbosity = Error;

Node::Node()
{
	treeName = nodeName;
    enabled = true;
	externalDisplay = false;
	drawResults = false;
}

Node::~Node()
{
    NodeManager::getInstance().onNodeRecompile(this);
}
void
Node::getInputs()
{

}
Node*
Node::addChild(Node* child)
{
    if (!child)
        return child;
    if(messageCallback)
        child->messageCallback = messageCallback;


    int count = children.get<NodeName>().count(child->nodeName);

	std::string prevTreeName = child->fullTreeName;
    child->setParent(fullTreeName, GetObjectId());
	child->setTreeName(child->nodeName + "-" + boost::lexical_cast<std::string>(count));
	
	// Notify the node manager of the tree name
	NodeManager::getInstance().updateTreeName(child, prevTreeName);

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
    return NodeManager::getInstance().getNode(itr->id);
}


Node*
Node::getChild(const int& index)
{
    auto itr = children.get<0>()[index];
	return NodeManager::getInstance().getNode(itr.id);
}

Node*
Node::getChild(const ObjectId& id)
{
	return NodeManager::getInstance().getNode(id);
}

boost::shared_ptr<Parameter> 
Node::getParameter(int idx)
{
	return parameters[idx];
}

boost::shared_ptr<Parameter> 
Node::getParameter(const std::string& name)
{
	for (int i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->name == name)
			return parameters[i];
	}
	return boost::shared_ptr<Parameter>();
}
std::vector<std::string> Node::listParameters()
{
	std::vector<std::string> paramList;
	for (int i = 0; i < parameters.size(); ++i)
	{
		paramList.push_back(parameters[i]->name);
	}
	return paramList;
}
std::vector<std::string> Node::listInputs()
{
	std::vector<std::string> paramList;
	for (int i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->type & Parameter::Input)
			paramList.push_back(parameters[i]->name);
	}
	return paramList;
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
    for(auto it = children.begin(); it != children.end(); ++it)
    {
        if(it->id == childId)
            children.erase(it);
    }

}

void
Node::removeChild(const std::string &name)
{
    auto itr = children.get<NodeName>().find(name);
    if(itr != children.get<NodeName>().end())
        children.get<NodeName>().erase(itr);

}

cv::cuda::GpuMat
Node::process(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    if(img.empty() && SkipEmpty())
    {

    }else
    {
        try
        {
            boost::posix_time::ptime start, end;
            {
                //boost::recursive_mutex::scoped_lock lock(mtx);
                start = boost::posix_time::microsec_clock::universal_time();
                if(debug_verbosity <= Status)
                {
                    log(Status, "Start: " + fullTreeName);
                }
                if(enabled)
                {
                    std::vector<boost::recursive_mutex::scoped_lock> locks;
                    for(int i = 0; i < parameters.size(); ++i)
                    {
                        locks.push_back(boost::recursive_mutex::scoped_lock(parameters[i]->mtx));
                    }
                    img = doProcess(img, stream);

                }
                end = boost::posix_time::microsec_clock::universal_time();
            }
            if(debug_verbosity <= Status)
            {
                log(Status, "End:   " + fullTreeName);
            }
            auto delta =  end - start;
            processingTime = delta.total_milliseconds();
        }catch(cv::Exception &err)
        {
            log(Error, err.what());
        }catch(std::exception &err)
        {
            log(Error, err.what());
        }
    }
    try
    {
        if(children.size() == 0)
            return img;
        int idx = 0;
        cv::cuda::GpuMat childResults;
        if(!img.empty())
            img.copyTo(childResults,stream);
        for (auto it = children.begin(); it != children.end(); ++it, ++idx)
        {
            ObjectId id = it->id;
            auto child = getChild(id);
            if(child)
                childResults = child->process(childResults, stream);
            else
                log(Error, "Null child with idx: " + boost::lexical_cast<std::string>(idx) +
                    " id: " + boost::lexical_cast<std::string>(id.m_ConstructorId) +
                    " " + boost::lexical_cast<std::string>(id.m_PerTypeId));
        }
        // So here is the debate of is a node's output the output of it, or the output of its children....
        // img = childResults;
    }catch(cv::Exception &err)
    {
        log(Error, err.what());
    }catch(std::exception &err)
    {
        log(Error, err.what());
    }catch(boost::exception &err)
    {
        log(Error, "Boost exception");
    }catch(...)
    {
        log(Error, "Unknown exception");
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
    catch (cv::Exception &err)
	{
        log(Error, err.what());
	}
}

cv::cuda::GpuMat
Node::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream stream )
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
Node::registerDisplayCallback(boost::function<void(cv::Mat, Node*)>& f)
{
    cpuDisplayCallback = f;
}

void
Node::registerDisplayCallback(boost::function<void(cv::cuda::GpuMat, Node*)>& f)
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
Node* Node::getParent()
{
    if(parentId.IsValid())
        return NodeManager::getInstance().getNode(parentId);
    if(parentName.size())
        return NodeManager::getInstance().getNode(parentName);
    return nullptr;
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
	SERIALIZE(fullTreeName);
    SERIALIZE(messageCallback);
    SERIALIZE(onUpdate);
    SERIALIZE(parentName);
    SERIALIZE(parentId);
    SERIALIZE(cpuDisplayCallback);
    SERIALIZE(gpuDisplayCallback);
    SERIALIZE(drawResults);
    SERIALIZE(externalDisplay);
    SERIALIZE(enabled);


}
std::vector<std::string>
Node::findType(Parameter::Ptr param)
{
    std::vector<Node*> nodes;
    NodeManager::getInstance().getAccessibleNodes(fullTreeName, nodes);
    return findType(param, nodes);
}

std::vector<std::string>
Node::findType(Loki::TypeInfo &typeInfo)
{
	std::vector<Node*> nodes;
	NodeManager::getInstance().getAccessibleNodes(fullTreeName, nodes);
    return findType(typeInfo, nodes);
}
std::vector<std::string>
Node::findType(Parameter::Ptr param, std::vector<Node*>& nodes)
{
    std::vector<std::string> output;

    for (int i = 0; i < nodes.size(); ++i)
    {
        if (nodes[i] == this)
            continue;
        for (int j = 0; j < nodes[i]->parameters.size(); ++j)
        {
            if (param->acceptsInput(nodes[i]->parameters[j]->typeInfo) && nodes[i]->parameters[j]->type & Parameter::Output)
                output.push_back(nodes[i]->parameters[j]->treeName);
        }
    }
    return output;
}

std::vector<std::string> 
Node::findType(Loki::TypeInfo &typeInfo, std::vector<Node*>& nodes)
{
	std::vector<std::string> output;

	for (int i = 0; i < nodes.size(); ++i)
	{
		if (nodes[i] == this)
			continue;
		for (int j = 0; j < nodes[i]->parameters.size(); ++j)
		{
            if (nodes[i]->parameters[j]->typeInfo == typeInfo && nodes[i]->parameters[j]->type & Parameter::Output)
				output.push_back(nodes[i]->parameters[j]->treeName);
		}
	}
	return output;
}
std::vector<std::vector<std::string>> 
Node::findCompatibleInputs()
{
	std::vector<std::vector<std::string>> output;
	for (int i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->type & Parameter::Input)
            output.push_back(findType(parameters[i]->typeInfo));
	}
	return output;
}
std::vector<std::string> Node::findCompatibleInputs(const std::string& paramName)
{
    std::vector<std::string> output;
    {
        auto param = getParameter(paramName);
        if(param)
            output = findType(param);
    }
    return output;
}
std::vector<std::string> Node::findCompatibleInputs(int paramIdx)
{
    return findCompatibleInputs(parameters[paramIdx]);
}

std::vector<std::string> Node::findCompatibleInputs(Parameter::Ptr param)
{
    return findType(param);
}


void
Node::setInputParameter(const std::string& sourceName, const std::string& inputName)
{
	for (int i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->name == inputName && parameters[i]->type & Parameter::Input)
			parameters[i]->setSource(sourceName);
	}
}

void
Node::setInputParameter(const std::string& sourceName, int inputIdx)
{
	int count = 0;
	for (int i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->type & Parameter::Input)
		{
			if (count == inputIdx)
				parameters[i]->setSource(sourceName);
			++count;
		}
	}
}
void
Node::setTreeName(const std::string& name)
{
	treeName = name;
    Node* parentPtr = NodeManager::getInstance().getNode(parentId);
	std::string fullTreeName_;
	if (parentPtr)
		fullTreeName_ = parentPtr->fullTreeName + "." + treeName;
	else
		fullTreeName_ = treeName;
	setFullTreeName(fullTreeName_);

}
void
Node::setFullTreeName(const std::string& name)
{
	for (int i = 0; i < parameters.size(); ++i)
	{
		parameters[i]->treeName = name + ":" + parameters[i]->name;
	}
	fullTreeName = name;
}

void
Node::setParent(const std::string& name, const ObjectId& parentId_)
{
    parentName = name;
    parentId = parentId_;
}
void 
Node::updateInputParameters()
{
	for (int i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->type & EagleLib::Parameter::Input)
		{
			parameters[i]->setSource("");
		}
	}
}
bool Node::SkipEmpty() const
{
    return true;
}
void Node::log(Verbosity level, const std::string &msg)
{

    if(messageCallback)
        messageCallback(level, msg, this);
    if(debug_verbosity > level && messageCallback)
        return;
    switch(level)
    {
    case Profiling:

    case Status:
        std::cout << "[ " << fullTreeName << " - STATUS ]" << msg << std::endl;
        break;
    case Warning:
        std::cout << "[ " << fullTreeName << " - WARNING ]" << msg << std::endl;
        break;
    case Error:
        std::cout << "[ " << fullTreeName << " - ERROR ]" << msg << std::endl;
        break;
    case Critical:
        std::cout << "[ " << fullTreeName << " - CRITICAL ]" << msg << std::endl;
        break;
    }
}
EventLoopNode::EventLoopNode():
    Node()
{

}

cv::cuda::GpuMat
EventLoopNode::process(cv::cuda::GpuMat &img, cv::cuda::Stream stream)
{
    service.run();
    return Node::process(img);
}

REGISTERCLASS(Node)
REGISTERCLASS(EventLoopNode)
