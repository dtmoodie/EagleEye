#include "nodes/Node.h"

#include <regex>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "Manager.h"
#include <boost/date_time.hpp>
#include <boost/thread.hpp>
#include <boost/log/trivial.hpp>
#include "logger.hpp"
#include <external_includes/cv_videoio.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <SystemTable.hpp>
#include <Events.h>

#include <EagleLib/NodeManager.h>
#include <EagleLib/DataStreamManager.h>
#include "../RuntimeObjectSystem/ISimpleSerializer.h"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
#include "remotery/lib/Remotery.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace EagleLib;
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

#define CATCH_MACRO                                                         \
    catch (boost::thread_resource_error& err)                               \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
catch (boost::thread_interrupted& err)                                      \
{                                                                           \
    NODE_LOG(error) << "Thread interrupted";                                \
    /* Needs to pass this back up to the chain to the processing thread.*/  \
    /* That way it knowns it needs to exit this thread */                   \
    throw err;                                                              \
}                                                                           \
catch (boost::thread_exception& err)                                        \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
    catch (cv::Exception &err)                                              \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
    catch (boost::exception &err)                                           \
{                                                                           \
    NODE_LOG(error) << "Boost error";                                       \
}                                                                           \
catch (std::exception &err)                                                 \
{                                                                           \
    NODE_LOG(error) << err.what();										    \
}                                                                           \
catch (...)                                                                 \
{                                                                           \
    NODE_LOG(error) << "Unknown exception";                                 \
}



namespace EagleLib
{
    struct NodeImpl
    {
        NodeImpl() :averageFrameTime(boost::accumulators::tag::rolling_window::window_size = 10)
        {
        }
		~NodeImpl()
		{
			for (auto itr : callbackConnections)
			{
				for (auto itr2 : itr.second)
				{
					itr2.disconnect();
				}
			}
			resetConnection.disconnect();
		}

        boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::rolling_mean> > averageFrameTime;
        std::vector<std::pair<time_t, int>> timings;
		boost::signals2::connection											resetConnection;
		std::map<EagleLib::Node*,std::vector<boost::signals2::connection>>							callbackConnections;
        boost::recursive_mutex mtx;
    };
}
NodeInfoRegisterer::NodeInfoRegisterer(const char* name, const char** hierarchy)
{
	
}
NodeInfoRegisterer::NodeInfoRegisterer(const char* nodeName, std::initializer_list<char const*> nodeInfo)
{
	std::vector<char const*> nodeInfoHierarchy(nodeInfo.begin(), nodeInfo.end());
	EagleLib::NodeManager::getInstance().RegisterNodeInfo(nodeName, nodeInfoHierarchy);
}
Node::Node():
    pImpl_(new NodeImpl())
{
	profile = false;
    enabled = true;
	externalDisplay = false;
    parent = nullptr;
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto signalHandler = table->GetSingleton<ISignalHandler>();
        auto signal = signalHandler->GetSignalSafe<boost::signals2::signal<void(void)>>("ResetSignal");
        boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
        pImpl_->resetConnection = signal->connect(boost::bind(&Node::reset, this));
    }
	rmt_hash = 0;
	NODE_LOG(trace) << " Constructor";
}

void Node::onParameterAdded()
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto signalHandler = table->GetSingleton<ISignalHandler>();
        auto signal = signalHandler->GetSignalSafe<boost::signals2::signal<void(Node*)>>("ParameterAdded");
        (*signal)(this);
    }
}

Parameters::Parameter* Node::addParameter(Parameters::Parameter::Ptr param)
{
    ParameteredObject::addParameter(param);
    param->SetTreeRoot(fullTreeName);
    onParameterAdded();
    return param.get();
}
void Node::onUpdate(cv::cuda::Stream* stream)
{
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto signalHandler = table->GetSingleton<ISignalHandler>();
        auto signal = signalHandler->GetSignalSafe<boost::signals2::signal<void(EagleLib::Node*)>>("NodeUpdated");
        (*signal)(this);
    }
}

Node::~Node()
{
    if(parent)
        parent->deregisterNotifier(this);
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
	auto& connections = pImpl_->callbackConnections[this];
	for (auto itr : connections)
	{
		itr.disconnect();
	}
	NODE_LOG(trace) << "Disconnected " <<connections.size() << " boost signals";
}
void Node::ClearProcessingTime()
{
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
    pImpl_->timings.clear();
}

void Node::EndProcessingTime()
{
    TIME;
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
	double total = pImpl_->timings[pImpl_->timings.size() - 1].first - pImpl_->timings[0].first;
	if (profile)
	{
		std::stringstream ss;
		for (int i = 1; i < pImpl_->timings.size(); ++i)
		{
			ss << pImpl_->timings[i - 1].second << "," << pImpl_->timings[i].second << "(" << pImpl_->timings[i].first - pImpl_->timings[i - 1].first << ")";
		}
		ss << " Total: " << total;
		NODE_LOG(trace) << ss.str();
	}
    pImpl_->averageFrameTime(total);
}

void Node::Clock(int line_num)
{
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
    pImpl_->timings.push_back(std::make_pair(clock(), line_num));
}
double Node::GetProcessingTime() const
{
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
    return boost::accumulators::rolling_mean(pImpl_->averageFrameTime);
}
std::vector<std::pair<time_t, int>> Node::GetProfileTimings() const
{
	boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
	return pImpl_->timings;
}
void Node::reset()
{
	NODE_LOG(trace);
	Init(false);
}

void Node::updateParent()
{
	NODE_LOG(trace);
    if(parent)
        parent->registerNotifier(this);
}

void
Node::getInputs()
{
	NODE_LOG(trace);
}
Node::Ptr
Node::addChild(Node* child)
{
	NODE_LOG(trace);
    return addChild(Node::Ptr(child));
}
Node::Ptr
Node::addChild(Node::Ptr child)
{
	NODE_LOG(trace);
    if (child == nullptr)
        return child;
    if(std::find(children.begin(), children.end(), child) != children.end())
        return child;
    int count = 0;
    for(size_t i = 0; i < children.size(); ++i)
    {
        if(children[i]->nodeName == child->nodeName)
            ++count;
    }
	
	child->SetDataStream(GetDataStream());
    child->setParent(this);
    child->setTreeName(child->nodeName + "-" + boost::lexical_cast<std::string>(count));
    children.push_back(child);
	BOOST_LOG_TRIVIAL(trace) << "[ " << fullTreeName << " ]" << " Adding child " << child->treeName;
    return child;
}

Node::Ptr
Node::getChild(const std::string& treeName)
{
	NODE_LOG(trace);
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
	NODE_LOG(trace);
    return children[index];
}
void
Node::swapChildren(int idx1, int idx2)
{
	NODE_LOG(trace);
    std::iter_swap(children.begin() + idx1, children.begin() + idx2);
}

void
Node::swapChildren(const std::string& name1, const std::string& name2)
{
	NODE_LOG(trace);
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
	NODE_LOG(trace);
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
	NODE_LOG(trace);
    std::vector<Node*> nodes;
    if(parent)
        parent->getNodesInScope(nodes);
    return nodes;
}
Node*
Node::getNodeInScope(const std::string& name)
{
	NODE_LOG(trace);
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
	// Perhaps not thread safe?
	NODE_LOG(trace);
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

/*Parameters::Parameter::Ptr Node::getParameter(int idx)
{
	NODE_LOG(trace);
	if (idx < parameters.size())
		return parameters[idx];
	else
		return Parameters::Parameter::Ptr();
}

Parameters::Parameter::Ptr Node::getParameter(const std::string& name)
{
	NODE_LOG(trace);
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		if (parameters[i]->GetName() == name)
			return parameters[i];
	}
	return Parameters::Parameter::Ptr();
}
*/
std::vector<std::string> Node::listParameters()
{
	NODE_LOG(trace);
	std::vector<std::string> paramList;
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		paramList.push_back(parameters[i]->GetName());
	}
	return paramList;
}
std::vector<std::string> Node::listInputs()
{
	NODE_LOG(trace);
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
	NODE_LOG(trace);

    // TODO tree structure parsing and correct directing of the search
    // Find the common base between this node and treeName


    return nullptr;
}

void
Node::removeChild(Node::Ptr node)
{
	NODE_LOG(trace);
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
	NODE_LOG(trace);
    children.erase(children.begin() + idx);
}

void
Node::removeChild(const std::string &name)
{
	NODE_LOG(trace);
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
	//rmt_ScopedCPUSample(process);
	
    if(boost::this_thread::interruption_requested())
        return img;
    ui_collector::setNode(this);
    
    if(img.empty() && SkipEmpty())
    {
		NODE_LOG(trace) << " Skipped due to empty input";
    }else
    {
		try
		{
			// Used for debugging which nodes have started, thus if a segfault occurs you can know which node caused it
			NODE_LOG(trace) << " process enabled: " << enabled;
			if (enabled)
			{
				ClearProcessingTime();
				boost::recursive_mutex::scoped_lock lock(mtx);

				// Do I lock each parameters mutex or do I just lock each node?
				// I should only lock each node, but then I need to make sure the UI keeps track of the node
				// to access the node's mutex while accessing a parameter, for now this works though.
			    /*std::vector<boost::recursive_mutex::scoped_lock> locks;
				for (size_t i = 0; i < parameters.size(); ++i)
				{
					locks.push_back(boost::recursive_mutex::scoped_lock(parameters[i]->mtx));
				}*/
				TIME
				_rmt_BeginCPUSample(fullTreeName.c_str(), &rmt_hash);
				_rmt_BeginCUDASample(fullTreeName.c_str(), &rmt_cuda_hash, cv::cuda::StreamAccessor::getStream(stream));
				img = doProcess(img, stream);
				rmt_EndCPUSample();
				rmt_EndCUDASample(cv::cuda::StreamAccessor::getStream(stream));
                EndProcessingTime();
			}
			NODE_LOG(debug) << "End:   " << fullTreeName;
		}CATCH_MACRO
    }
	try
	{
		if (children.size() == 0)
			return img;

		cv::cuda::GpuMat childResult;
		if (!img.empty())
			img.copyTo(childResult, stream);
		NODE_LOG(trace) << " Executing " << children.size() << " child nodes";
		std::vector<Node::Ptr>  children_;
		children_.reserve(children.size());
		{
			// Prevents adding of children while running, debatable how much this is needed
			boost::recursive_mutex::scoped_lock lock(mtx);
			for (int i = 0; i < children.size(); ++i)
			{
				children_.push_back(children[i]);
			}
		}
		for (size_t i = 0; i < children_.size(); ++i)
		{
			if (children_[i] != nullptr)
			{
				try
				{
					childResult = children_[i]->process(childResult, stream);
				}CATCH_MACRO
			}
			else
			{
                ui_collector::setNode(this);
				NODE_LOG(error) << "Null child with idx: " + boost::lexical_cast<std::string>(i);
			}
		}
        ui_collector::setNode(this);
		// So here is the debate of is a node's output the output of it, or the output of its children....
		// img = childResults;
    }CATCH_MACRO;
    ui_collector::setNode(nullptr);
	
    return img;
}
void Node::SetDataStream(std::shared_ptr<DataStream> stream_)
{
	if (_dataStream)
	{
		NODE_LOG(debug) << "Updating stream manager to a new manager";
	}	
	else
	{
		NODE_LOG(debug) << "Setting stream manager";
	}
    _dataStream = stream_;
	for (auto& child : children)
	{
		child->SetDataStream(_dataStream);
	}
}
std::shared_ptr<DataStream> Node::GetDataStream()
{
	if (parent && _dataStream == nullptr)
	{
		SetDataStream(parent->GetDataStream());
	}
	if (parent == nullptr && _dataStream == nullptr)
	{
        _dataStream = DataStreamManager::instance()->create_stream();
	}	
	return _dataStream;
}
cv::cuda::GpuMat
Node::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream )
{
	NODE_LOG(trace);
    return img;
}



void
Node::registerDisplayCallback(boost::function<void(cv::Mat, Node*)>& f)
{
	NODE_LOG(trace);
    //cpuDisplayCallback = f;
}

void
Node::registerDisplayCallback(boost::function<void(cv::cuda::GpuMat, Node*)>& f)
{
	NODE_LOG(trace);
	//gpuDisplayCallback = f;
}

void
Node::spawnDisplay()
{
	NODE_LOG(trace);
	cv::namedWindow(treeName);
	externalDisplay = true;
}
void
Node::killDisplay()
{
	NODE_LOG(trace);
	if (externalDisplay)
		cv::destroyWindow(treeName);
}
std::string
Node::getName() const
{
	NODE_LOG(trace);
    return nodeName;
}
std::string
Node::getTreeName() const
{
	NODE_LOG(trace);
    return treeName;
}
Node*
Node::getParent()
{
	NODE_LOG(trace);
    return parent;
}


Node*
Node::swap(Node* other)
{
    // By moving ownership of all parameters to the new node, all
	NODE_LOG(trace);
    return other;
}
void
Node::Init(bool firstInit)
{
    ui_collector::setNode(this);
	NODE_LOG(trace);
    IObject::Init(firstInit);
	if (!firstInit)
	{
		for (auto& param : parameters)
		{
			//pImpl_->callbackConnections[this].push_back(param->RegisterNotifier(boost::bind(&Node::onUpdate, this, _1)));
            RegisterParameterCallback(param.get(), boost::bind(&Node::onUpdate, this, _1));
		}
	}
}

void
Node::Init(const std::string &configFile)
{
    ui_collector::setNode(this);
	NODE_LOG(trace);
}
/*void Node::RegisterParameterCallback(int idx, boost::function<void(cv::cuda::Stream*)> callback)
{
	NODE_LOG(trace);
	auto param = getParameter(idx);
	if (param)
	{
        boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
		pImpl_->callbackConnections[this].push_back(param->RegisterNotifier(callback));
	}
}
void Node::RegisterParameterCallback(const std::string& name, boost::function<void(cv::cuda::Stream*)> callback)
{
	NODE_LOG(trace);
	auto param = getParameter(name);
	if (param)
	{
        boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
		pImpl_->callbackConnections[this].push_back(param->RegisterNotifier(callback));
	}
}*/
void Node::RegisterSignalConnection(boost::signals2::connection connection)
{
	boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
	pImpl_->callbackConnections[this].push_back(connection);
}

void
Node::Init(const cv::FileNode& configNode)
{
    ui_collector::setNode(this);
	NODE_LOG(trace) << " Initializing from file";
    configNode["NodeName"] >> nodeName;
    configNode["NodeTreeName"] >> treeName;
    configNode["FullTreeName"] >> fullTreeName;
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
            ui_collector::setNode(this);
		}
		else
		{
			NODE_LOG(error) << "No node found with the name " << name;
		}
    }
    cv::FileNode paramNode = configNode["Parameters"];
    for (size_t i = 0; i < parameters.size(); ++i)
    {
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

            }
            else
            {
                if (parameters[i]->type & Parameters::Parameter::Control)
                    Parameters::Persistence::cv::DeSerialize(&paramNode, parameters[i].get());
            }
        }
        catch (cv::Exception &e)
        {
            BOOST_LOG_TRIVIAL(error) << "Deserialization failed for " << parameters[i]->GetName() << " with type " << parameters[i]->GetTypeInfo().name() << std::endl;
        }
    }
}

void
Node::Serialize(ISimpleSerializer *pSerializer)
{
	NODE_LOG(trace) << " Serializing";
    ParameteredObject::Serialize(pSerializer);
    SERIALIZE(children);
    SERIALIZE(treeName);
    SERIALIZE(nodeName);
	SERIALIZE(fullTreeName);
    SERIALIZE(parent);
    SERIALIZE(externalDisplay);
    SERIALIZE(enabled);
    SERIALIZE(pImpl_);
}

void
Node::Serialize(cv::FileStorage& fs)
{
	NODE_LOG(trace) << " Serializing to file";
    if(fs.isOpened())
    {
        fs << "NodeName" << nodeName;
        fs << "NodeTreeName" << treeName;
        fs << "FullTreeName" << fullTreeName;
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
						NODE_LOG(warning) << e.what();
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
	NODE_LOG(trace);
    std::vector<Node*> nodes;
    getNodesInScope(nodes);
    return findType(param, nodes);
}

std::vector<std::string>
Node::findType(Loki::TypeInfo typeInfo)
{
	NODE_LOG(trace);
    std::vector<Node*> nodes;
    getNodesInScope(nodes);
    return findType(typeInfo, nodes);
}
std::vector<std::string>
Node::findType(Parameters::Parameter::Ptr param, std::vector<Node*>& nodes)
{
	NODE_LOG(trace);
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
	NODE_LOG(trace);
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
	NODE_LOG(trace);
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
	NODE_LOG(trace);
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
	NODE_LOG(trace);
    return findCompatibleInputs(parameters[paramIdx]);
}

std::vector<std::string> Node::findCompatibleInputs(Parameters::Parameter::Ptr param)
{
	NODE_LOG(trace);
    return findType(param);
}
std::vector<std::string> Node::findCompatibleInputs(Loki::TypeInfo& type)
{
	NODE_LOG(trace);
	return findType(type);
}
std::vector<std::string> Node::findCompatibleInputs(Parameters::InputParameter::Ptr param)
{
	NODE_LOG(trace);
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
	NODE_LOG(trace);
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
	NODE_LOG(trace);
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
	NODE_LOG(trace);
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
	NODE_LOG(trace);
    for (size_t i = 0; i < parameters.size(); ++i)
	{
		parameters[i]->SetTreeRoot(name);
	}
	fullTreeName = name;
}

void
Node::setParent(Node* parent_)
{
	NODE_LOG(trace);
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
	NODE_LOG(trace);
    parent = static_cast<Node*>(ptr);
}

void 
Node::updateInputParameters()
{
	NODE_LOG(trace);
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

