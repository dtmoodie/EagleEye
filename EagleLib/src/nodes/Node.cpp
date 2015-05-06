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
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core")
#endif



#endif
Verbosity Node::debug_verbosity = Error;

Node::Node()
{
	treeName = nodeName;
    enabled = true;
	externalDisplay = false;
	drawResults = false;
    parent = nullptr;
}

Node::~Node()
{
    if(parent)
        parent->deregisterNotifier(this);
}
void
Node::getInputs()
{

}
Node::Ptr
Node::addChild(Node* child)
{
    return addChild(Node::Ptr(child));
}
Node::Ptr
Node::addChild(Node::Ptr child)
{
    if (child == nullptr)
        return child;
    if(std::find(children.begin(), children.end(), child) != children.end())
        return child;
    if(messageCallback)
        child->messageCallback = messageCallback;
    int count = 0;
    for(int i = 0; i < children.size(); ++i)
    {
        if(children[i]->nodeName == child->nodeName)
            ++count;
    }
    child->setParent(this);
    child->setTreeName(child->nodeName + "-" + boost::lexical_cast<std::string>(count));
    children.push_back(child);
    return child;
}

Node::Ptr
Node::getChild(const std::string& treeName)
{
    for(int i = 0; i < children.size(); ++i)
    {
        if(children[i]->treeName == treeName)
            return children[i];
    }
    return Node::Ptr();
}


Node::Ptr
Node::getChild(const int& index)
{
    return children[index];
}
void
Node::swapChildren(int idx1, int idx2)
{
    std::iter_swap(children.begin() + idx1, children.begin() + idx2);
}

void
Node::swapChildren(const std::string& name1, const std::string& name2)
{
    auto itr1 = children.begin();
    auto itr2 = children.begin();
    for(; itr1 != children.begin(); ++itr1)
    {
        if((*itr1)->treeName == name1)
            break;
    }
    for(; itr2 != children.begin(); ++itr2)
    {
        if((*itr2)->treeName == name2)
            break;
    }
    if(itr1 != children.end() && itr2 != children.end())
        std::iter_swap(itr1,itr2);
}
void
Node::swapChildren(Node::Ptr child1, Node::Ptr child2)
{
    auto itr1 = std::find(children.begin(),children.end(), child1);
    if(itr1 == children.end())
        return;
    auto itr2 = std::find(children.begin(), children.end(), child2);
    if(itr2 == children.end())
        return;
    std::iter_swap(itr1,itr2);
}
std::vector<Node::Ptr>
Node::getNodesInScope()
{
    std::vector<Node::Ptr> nodes;
    if(parent)
        parent->getNodesInScope(nodes);
    return nodes;
}
Node*
Node::getNodeInScope(const std::string& name)
{
    // Check if this is a child node of mine, if not go up
    int ret = name.compare(0, fullTreeName.length(), fullTreeName);
    if(ret == 0)
    {
        // name is a child of current node, or is the current node
        if(fullTreeName.size() == name.size())
            return this;
        std::string childName = name.substr(fullTreeName.size() + 1);
        auto child = getChild(childName);
        if(child != nullptr)
            return child.get();
    }
    if(parent)
        return parent->getNodeInScope(name);
    return nullptr;
}

void
Node::getNodesInScope(std::vector<Node::Ptr>& nodes)
{
    nodes.insert(nodes.end(), children.begin(), children.end());
    if(parent)
        parent->getNodesInScope(nodes);
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
Node::removeChild(Node::Ptr node)
{
    for(auto itr = children.begin(); itr != children.end(); ++itr)
    {
        if(*itr == node)
        {
            children.erase(itr);
                    return;
        }
    }
}
void
Node::removeChild(int idx)
{
    children.erase(children.begin() + idx);
}

void
Node::removeChild(const std::string &name)
{
    for(auto itr = children.begin(); itr != children.end(); ++itr)
    {
        if((*itr)->treeName == name)
        {
            children.erase(itr);
            return;
        }
    }
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
                    boost::recursive_mutex::scoped_lock lock(mtx);
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

        cv::cuda::GpuMat childResults;
        if(!img.empty())
            img.copyTo(childResults,stream);
        boost::recursive_mutex::scoped_lock lock(mtx);
        for(int i = 0; i < children.size(); ++i)
        {
            if(children[i] != nullptr)
            {
                childResults = children[i]->process(childResults, stream);
            }else
            {
                log(Error, "Null child with idx: " + boost::lexical_cast<std::string>(i));
            }
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
Node*
Node::getParent()
{
    return parent;
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
    IObject::Init(firstInit);
    if(parent)
        parent->registerNotifier(this);
}

void
Node::Init(const std::string &configFile)
{

}

void
Node::Init(const cv::FileNode& configNode)
{
    configNode["NodeName"] >> nodeName;
    configNode["NodeTreeName"] >> treeName;
    configNode["FullTreeName"] >> fullTreeName;
    configNode["DrawResults"] >> drawResults;
    configNode["Enabled"] >> enabled;
    configNode["ExternalDisplay"] >> externalDisplay;
    cv::FileNode childrenFS = configNode["Children"];
    int childCount = (int)childrenFS["Count"];
    for(int i = 0; i < childCount; ++i)
    {
        cv::FileNode childNode = childrenFS["Node-" + boost::lexical_cast<std::string>(i)];
        std::string name = (std::string)childNode["NodeName"];
        auto node = NodeManager::getInstance().addNode(name);
        node->Init(childNode);
        addChild(node);
    }
    cv::FileNode paramNode =  configNode["Parameters"];
    for(int i = 0; i < parameters.size(); ++i)
    {
        parameters[i]->Init(paramNode);
    }
    // Figure out parameter loading :/  Need some kind of factory for all of the parameter types
}

void
Node::Serialize(ISimpleSerializer *pSerializer)
{
    IObject::Serialize(pSerializer);
    SERIALIZE(parameters);
    SERIALIZE(children);
    SERIALIZE(treeName);
    SERIALIZE(nodeName);
	SERIALIZE(fullTreeName);
    SERIALIZE(messageCallback);
    SERIALIZE(onUpdate);
    SERIALIZE(parent);
    SERIALIZE(cpuDisplayCallback);
    SERIALIZE(gpuDisplayCallback);
    SERIALIZE(drawResults);
    SERIALIZE(externalDisplay);
    SERIALIZE(enabled);
}
void
Node::Serialize(cv::FileStorage& fs)
{
    if(fs.isOpened())
    {
        fs << "NodeName" << nodeName;
        fs << "NodeTreeName" << treeName;
        fs << "FullTreeName" << fullTreeName;
        fs << "DrawResults" << drawResults;
        fs << "Enabled" << enabled;
        fs << "ExternalDisplay" << externalDisplay;
        fs << "Children" << "{";
        fs << "Count" << (int)children.size();
        for(int i = 0; i < children.size(); ++i)
        {
            fs << "Node-" + boost::lexical_cast<std::string>(i) << "{";
            children[i]->Serialize(fs);
            fs << "}";
        }
        fs << "}"; // end children

        fs << "Parameters" << "{";
        for(int i = 0; i < parameters.size(); ++i)
        {
            if(parameters[i]->type & Parameter::Control)
            {
                parameters[i]->Serialize(fs);
            }
        }
        fs << "}"; // end parameters

    }
}

std::vector<std::string>
Node::findType(Parameter::Ptr param)
{
    std::vector<Node::Ptr> nodes;
    getNodesInScope(nodes);
    return findType(param, nodes);
}

std::vector<std::string>
Node::findType(Loki::TypeInfo &typeInfo)
{
    std::vector<Node::Ptr> nodes;
    getNodesInScope(nodes);
    return findType(typeInfo, nodes);
}
std::vector<std::string>
Node::findType(Parameter::Ptr param, std::vector<Node::Ptr>& nodes)
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
Node::findType(Loki::TypeInfo &typeInfo, std::vector<Ptr> &nodes)
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
	std::string fullTreeName_;
    if (parent == nullptr)
        fullTreeName_ = treeName;
	else
        fullTreeName_ = parent->fullTreeName + "." + treeName;
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
Node::setParent(Node* parent_)
{
    if(parent)
    {
        parent->deregisterNotifier(this);
    }
    parent = parent_;
    parent->registerNotifier(this);
}
void
Node::updateObject(IObject* ptr)
{
    parent = static_cast<Node*>(ptr);
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
