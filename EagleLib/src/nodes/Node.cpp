#include "nodes/Node.h"

#include <regex>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "Manager.h"
#include <boost/date_time.hpp>
#include <boost/thread.hpp>
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
boost::signals2::signal<void(void)> Node::resetSignal = boost::signals2::signal<void(void)>();
Node::Node():
    averageFrameTime(boost::accumulators::tag::rolling_window::window_size = 10)
{
	treeName = nodeName;
	profile = false;
    enabled = true;
	externalDisplay = false;
	drawResults = false;
    parent = nullptr;
	nodeType = eVirtual;
    onParameterAdded.reset(new boost::signals2::signal<void(void)>);
	resetConnection = resetSignal.connect(boost::bind(&Node::reset, this));
}

Node::~Node()
{
    if(parent)
        parent->deregisterNotifier(this);
	for (int i = 0; i < callbackConnections.size(); ++i)
	{
		callbackConnections[i].disconnect();
	}
}
void Node::reset()
{
	Init(false);
}

void Node::updateParent()
{
    if(parent)
        parent->registerNotifier(this);
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
    for(size_t i = 0; i < children.size(); ++i)
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
    for(size_t i = 0; i < children.size(); ++i)
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
std::vector<Node *> Node::getNodesInScope()
{
    std::vector<Node*> nodes;
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
Node::getNodesInScope(std::vector<Node *> &nodes)
{
    // First travel to the root node

    if(nodes.size() == 0)
    {
        Node* node = this;
        while(node->parent != nullptr)
        {
            node = node->parent;
        }
        nodes.push_back(node);
        node->getNodesInScope(nodes);
        return;
    }
    nodes.push_back(this);
    for(size_t i = 0; i < children.size(); ++i)
    {
        if(children[i] != nullptr)
            children[i]->getNodesInScope(nodes);
    }
}

Parameters::Parameter::Ptr Node::getParameter(int idx)
{
	if (idx < parameters.size())
		return parameters[idx];
	else
		return Parameters::Parameter::Ptr();
}

Parameters::Parameter::Ptr Node::getParameter(const std::string& name)
{
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->GetName() == name)
			return parameters[i];
	}
	return Parameters::Parameter::Ptr();
}
std::vector<std::string> Node::listParameters()
{
	std::vector<std::string> paramList;
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		paramList.push_back(parameters[i]->GetName());
	}
	return paramList;
}
std::vector<std::string> Node::listInputs()
{
	std::vector<std::string> paramList;
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->type & Parameters::Parameter::Input)
			paramList.push_back(parameters[i]->GetName());
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
Node::process(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(boost::this_thread::interruption_requested())
        return img;
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
                // Used for debugging which nodes have started, thus if a segfault occurs you can know which node caused it
                if(debug_verbosity <= Status)
                {
                    log(Status, "Start: " + fullTreeName);
                }
                if(enabled)
                {
                    boost::recursive_mutex::scoped_lock lock(mtx);
                    // Do I lock each parameters mutex or do I just lock each node?
                    // I should only lock each node, but then I need to make sure the UI keeps track of the node
                    // to access the node's mutex while accessing a parameter, for now this works though.
                    std::vector<boost::recursive_mutex::scoped_lock> locks;
                    for(size_t i = 0; i < parameters.size(); ++i)
                    {
						locks.push_back(boost::recursive_mutex::scoped_lock(parameters[i]->mtx));
                    }
                    if(profile)
                    {
                        timings.clear();
                        TIME
                    }
                    img = doProcess(img, stream);
                    if(profile)
                    {
                        TIME
                        std::stringstream time;
						if (timings.size() > 2)
						{
							time << "Start: " << timings[1].first - timings[0].first << " ";
							for (size_t i = 2; i < timings.size() - 1; ++i)
							{
								time << "(" << timings[i - 1].second << "," << timings[i].second << "," << timings[i].first - timings[i - 1].first << ")";
							}
							time << "End: " << timings[timings.size() - 1].first - timings[timings.size() - 2].first;
							log(Profiling, time.str());
						}
                    }
                }
                end = boost::posix_time::microsec_clock::universal_time();
            }
            if(debug_verbosity <= Status)
            {
                log(Status, "End:   " + fullTreeName);
            }
            auto delta =  end - start;
            averageFrameTime(delta.total_milliseconds());
            processingTime = boost::accumulators::rolling_mean(averageFrameTime);
        CATCH_MACRO
    }
    try
    {
        if(children.size() == 0)
            return img;

        cv::cuda::GpuMat* childResult = childResults.getFront();
        if(!img.empty())
            img.copyTo(*childResult,stream);

        std::vector<Node::Ptr>  children_;
        children_.reserve(children.size());
        {
            // Prevents adding of children while running, debatable how much this is needed
            boost::recursive_mutex::scoped_lock lock(mtx);
            for(int i = 0; i < children.size(); ++i)
            {
                children_.push_back(children[i]);
            }
        }
        for(size_t i = 0; i < children_.size(); ++i)
        {
            if(children_[i] != nullptr)
            {
                try
                {
                *childResult = children_[i]->process(*childResult, stream);
                CATCH_MACRO
            }else
            {
                log(Error, "Null child with idx: " + boost::lexical_cast<std::string>(i));
            }
        }
        // So here is the debate of is a node's output the output of it, or the output of its children....
        // img = childResults;
    CATCH_MACRO
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
Node::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream )
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
}

void
Node::Init(const std::string &configFile)
{

}
void Node::RegisterParameterCallback(int idx, boost::function<void(void)> callback)
{
	auto param = getParameter(idx);
	if (param)
	{
		callbackConnections.push_back(param->RegisterNotifier(callback));
	}
}
void Node::RegisterParameterCallback(const std::string& name, boost::function<void(void)> callback)
{
	auto param = getParameter(name);
	if (param)
	{
		callbackConnections.push_back(param->RegisterNotifier(callback));
	}
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
		if (node != nullptr)
		{
			addChild(node);
			node->Init(childNode);
		}
		else
		{
			log(Error, "No node found with the name " + name);
		}
    }
    cv::FileNode paramNode =  configNode["Parameters"];
    for(size_t i = 0; i < parameters.size(); ++i)
    {
		// TODO
		try
		{
			if (parameters[i]->type & Parameters::Parameter::Input)
			{
				auto node = paramNode[parameters[i]->GetName()];
				auto inputName = (std::string)node["InputParameter"];
				if (inputName.size())
				{
					auto idx = inputName.find(':');
					auto nodeName = inputName.substr(0, idx);
					auto paramName = inputName.substr(idx + 1);
					auto nodes = getNodesInScope();
					auto node = getNodeInScope(nodeName);
					if (node)
					{
						auto param = node->getParameter(paramName);
						if (param)
						{
							auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(parameters[i]);
							inputParam->SetInput(param);
						}
					}
				}

			}else
			{
				if (parameters[i]->type & Parameters::Parameter::Control)
					Parameters::Persistence::cv::DeSerialize(&paramNode, parameters[i].get());
			}
		}
		catch (cv::Exception &e)
		{
			std::cout << "Deserialization failed for " << parameters[i]->GetName() << " with type " << parameters[i]->GetTypeInfo().name() << std::endl;
		}
        //parameters[i]->Init(paramNode);
        //parameters[i]->update();
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
    SERIALIZE(onParameterAdded);
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
        for(size_t i = 0; i < children.size(); ++i)
        {
            fs << "Node-" + boost::lexical_cast<std::string>(i) << "{";
            children[i]->Serialize(fs);
            fs << "}";
        }
        fs << "}"; // end children

        fs << "Parameters" << "{";
        for(size_t i = 0; i < parameters.size(); ++i)
        {
			if (parameters[i]->type & Parameters::Parameter::Input)
			{
				auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(parameters[i]);
				if (inputParam)
				{
					auto input = inputParam->GetInput();
					if (input)
					{
						fs << parameters[i]->GetName().c_str() << "{";
						fs << "TreeName" << parameters[i]->GetTreeName();
						fs << "InputParameter" << input->GetTreeName();
						fs << "Type" << parameters[i]->GetTypeInfo().name();
						auto toolTip = parameters[i]->GetTooltip();
						if (toolTip.size())
							fs << "ToolTip" << toolTip;
						fs << "}";
					}
				}
			}
			else
			{
				if (parameters[i]->type & Parameters::Parameter::Control)
				{
					// TODO
					try
					{
						Parameters::Persistence::cv::Serialize(&fs, parameters[i].get());
					}
					catch (cv::Exception &e)
					{
						continue;
					}
				}
			}            
        }
        fs << "}"; // end parameters

    }
}

std::vector<std::string>
Node::findType(Parameters::Parameter::Ptr param)
{
    std::vector<Node*> nodes;
    getNodesInScope(nodes);
    return findType(param, nodes);
}

std::vector<std::string>
Node::findType(Loki::TypeInfo &typeInfo)
{
    std::vector<Node*> nodes;
    getNodesInScope(nodes);
    return findType(typeInfo, nodes);
}
std::vector<std::string>
Node::findType(Parameters::Parameter::Ptr param, std::vector<Node*>& nodes)
{
    std::vector<std::string> output;
	auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(param);
	if (inputParam)
	{
		for (size_t i = 0; i < nodes.size(); ++i)
		{
			if (nodes[i] == this)
				continue;
			for (size_t j = 0; j < nodes[i]->parameters.size(); ++j)
			{

				if (nodes[i]->parameters[j]->type & Parameters::Parameter::Output)
				{
					if (inputParam->AcceptsInput(nodes[i]->parameters[j]))
					{
						output.push_back(nodes[i]->parameters[j]->GetTreeName());
					}
					
				}

			}
		}
	}
    return output;
}

std::vector<std::string> 
Node::findType(Loki::TypeInfo &typeInfo, std::vector<Node*> &nodes)
{
	std::vector<std::string> output;

    for (size_t i = 0; i < nodes.size(); ++i)
	{
		if (nodes[i] == this)
			continue;
        for (size_t j = 0; j < nodes[i]->parameters.size(); ++j)
		{
			if (nodes[i]->parameters[j]->type & Parameters::Parameter::Output)
			{
				if (nodes[i]->parameters[j]->GetTypeInfo() == typeInfo)
				{
					output.push_back(nodes[i]->parameters[j]->GetTreeName());
				}
			}
			
		}
	}
	return output;
}
std::vector<std::vector<std::string>> 
Node::findCompatibleInputs()
{
	std::vector<std::vector<std::string>> output;
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->type & Parameters::Parameter::Input)
            output.push_back(findType(parameters[i]->GetTypeInfo()));
	}
	return output;
}
std::vector<std::string> Node::findCompatibleInputs(const std::string& paramName)
{
    std::vector<std::string> output;
    {
        auto param = Node::getParameter(paramName);
        if(param)
            output = findType(param);
    }
    return output;
}
std::vector<std::string> Node::findCompatibleInputs(int paramIdx)
{
    return findCompatibleInputs(parameters[paramIdx]);
}

std::vector<std::string> Node::findCompatibleInputs(Parameters::Parameter::Ptr param)
{
    return findType(param);
}
std::vector<std::string> Node::findCompatibleInputs(Loki::TypeInfo& type)
{
	return findType(type);
}
std::vector<std::string> Node::findCompatibleInputs(Parameters::InputParameter::Ptr param)
{
	std::vector<Node*> nodes;
	std::vector<std::string> output;
	getNodesInScope(nodes);
	for (int i = 0; i < nodes.size(); ++i)
	{
		for (int j = 0; j < nodes[i]->parameters.size(); ++j)
		{
			if (!(nodes[i]->parameters[j]->type & Parameters::Parameter::Input))
				if (param->AcceptsInput(nodes[i]->parameters[j]))
					output.push_back(nodes[i]->parameters[j]->GetTreeName());
		}
	}
	return output;
}


void
Node::setInputParameter(const std::string& sourceName, const std::string& inputName)
{
	auto param = getParameter(inputName);
	auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(param);
	if (inputParam)
	{
		inputParam->SetInput(sourceName);
	}
}

void
Node::setInputParameter(const std::string& sourceName, int inputIdx)
{
	auto param = getParameter(inputIdx);
	auto inputParam = std::dynamic_pointer_cast<Parameters::InputParameter>(param);
	if (inputParam)
	{
		inputParam->SetInput(sourceName);
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
    for(size_t i = 0; i < children.size(); ++i)
    {
        children[i]->setTreeName(children[i]->treeName);
    }
}
void
Node::setFullTreeName(const std::string& name)
{
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		parameters[i]->SetTreeRoot(name);
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
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->type & Parameters::Parameter::Input)
		{
			// TODO
			//parameters[i]->setSource("");
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


REGISTERCLASS(Node)
