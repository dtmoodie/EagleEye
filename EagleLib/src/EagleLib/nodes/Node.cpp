#ifndef OPENCV_FOUND
#define OPENCV_FOUND
#endif
#include "EagleLib/nodes/Node.h"
#include "EagleLib/nodes/NodeFactory.h"
#include <EagleLib/frame_grabber_base.h>
#include <EagleLib/DataStreamManager.h>
#include <EagleLib/rcc/external_includes/cv_videoio.hpp>
#include <EagleLib/rcc/SystemTable.hpp>
#include <EagleLib/utilities/GpuMatAllocators.h>
#include "EagleLib/Signals.h"
#include "EagleLib/profiling.h"

#include "../RuntimeObjectSystem/ISimpleSerializer.h"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Signals/Connection.hpp>
#include <MetaObject/Logging/Log.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/date_time.hpp>
#include <boost/thread.hpp>
#include <boost/log/trivial.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "remotery/lib/Remotery.h"
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <regex>
using namespace EagleLib;
using namespace EagleLib::Nodes;
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

#define CATCH_MACRO                                                         \
    catch(mo::ExceptionWithCallStack<cv::Exception>& e)                \
{                                                                           \
    NODE_LOG(error) << e.what() << "\n" << e.CallStack();                   \
}                                                                           \
    catch(mo::ExceptionWithCallStack<std::string>& e)                  \
{                                                                           \
    NODE_LOG(error) << std::string(e) << "\n" << e.CallStack();                   \
}                                                                           \
catch(mo::IExceptionWithCallStackBase& e)                              \
{                                                                           \
    NODE_LOG(error) << "Exception thrown with callstack: \n" << e.CallStack(); \
}                                                                           \
catch (boost::thread_resource_error& err)                                   \
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
    NODE_LOG(error) << err.what();                                            \
}                                                                           \
catch (...)                                                                 \
{                                                                           \
    NODE_LOG(error) << "Unknown exception";                                 \
}

Nodes::NodeInfo::NodeInfo(const char* name, std::initializer_list<char const*> nodeInfo):
    node_name(name)
{
    for (auto itr : nodeInfo)
    {
        node_hierarchy.push_back(itr);
    }
}

int Nodes::NodeInfo::GetInterfaceId() const
{
    return IID_NodeObject;
}
std::string Nodes::NodeInfo::GetObjectName() const
{
    return node_name;
}
std::string Nodes::NodeInfo::GetObjectTooltip() const
{
    return node_tooltip;
}
std::string Nodes::NodeInfo::GetObjectHelp() const
{
    return node_help;
}

std::string NodeInfo::Print() const
{
    return mo::IMetaObjectInfo::Print();
}

std::vector<const char*> Nodes::NodeInfo::GetNodeHierarchy() const
{
    return node_hierarchy;
}

std::vector<std::vector<std::string>> Nodes::NodeInfo::GetParentalDependencies() const
{
    return std::vector<std::vector<std::string>>();
}


std::vector<std::vector<std::string>> Nodes::NodeInfo::GetNonParentalDependencies() const
{
    return std::vector<std::vector<std::string>>();
}


std::vector<std::string> Nodes::NodeInfo::CheckDependentVariables(mo::IVariableManager* var_manager_) const
{
    return std::vector<std::string>();
}

namespace EagleLib
{
    namespace Nodes
    {
        struct NodeImpl
        {
            NodeImpl() :averageFrameTime(boost::accumulators::tag::rolling_window::window_size = 10)
            {
                //update_signal = nullptr;
                //g_update_signal = nullptr;
            }
            ~NodeImpl()
            {
                for (auto itr : callbackConnections)
                {
                    for (auto itr2 : itr.second)
                    {
                        //itr2.disconnect();
                        itr2.reset();
                    }
                }
            }

            boost::accumulators::accumulator_set<double, boost::accumulators::features<boost::accumulators::tag::rolling_mean> > averageFrameTime;
            std::vector<std::pair<time_t, int>>                                                     timings;
            std::shared_ptr<mo::Connection>                                                    resetConnection;
            std::map<EagleLib::Nodes::Node*, std::vector<std::shared_ptr<mo::Connection>>>     callbackConnections;
            std::map<EagleLib::Nodes::Node*, std::vector<std::shared_ptr<mo::Connection>>>        callbackConnections2;
            mo::TypedSignal<void(Nodes::Node*)>*                                                    update_signal;
            mo::TypedSignal<void(Nodes::Node*)>*                                                    g_update_signal;
            boost::recursive_mutex                                                                  mtx;
        };
    }    
}
Nodes::NodeInfoRegisterer::NodeInfoRegisterer(const char* name, const char** hierarchy)
{
    
}
Nodes::NodeInfoRegisterer::NodeInfoRegisterer(const char* nodeName, std::initializer_list<char const*> nodeInfo)
{
    std::vector<char const*> nodeInfoHierarchy(nodeInfo.begin(), nodeInfo.end());
    //EagleLib::NodeManager::getInstance().RegisterNodeInfo(nodeName, nodeInfoHierarchy);
}

Node::Node():
    pImpl_(new NodeImpl())
{
    
    enabled = true;
    externalDisplay = false;
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto signal_manager = table->GetSingleton<mo::RelayManager>();
        signal_manager->ConnectSlots(this, "reset");
    }
    rmt_hash = 0;
    LOG(trace) << " Constructor";
    _dataStream = nullptr;
}


Node::~Node()
{
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
    auto& connections = pImpl_->callbackConnections[this];
    for (auto itr : connections)
    {
        itr.reset();
    }
    auto itr = pImpl_->callbackConnections2.find(this);
    if(itr != pImpl_->callbackConnections2.end())
        pImpl_->callbackConnections2.erase(itr);
    LOG(trace) << "Disconnected " <<connections.size() << " boost signals";
}

void Node::ClearProcessingTime()
{
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
    pImpl_->timings.clear();
}

void Node::EndProcessingTime()
{
    
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
    double total = pImpl_->timings[pImpl_->timings.size() - 1].first - pImpl_->timings[0].first;
    
    std::stringstream ss;
    for (int i = 1; i < pImpl_->timings.size(); ++i)
    {
        ss << pImpl_->timings[i - 1].second << "," << pImpl_->timings[i].second << "(" << pImpl_->timings[i].first - pImpl_->timings[i - 1].first << ")";
    }
    ss << " Total: " << total;
    LOG(trace) << ss.str();
    
    pImpl_->averageFrameTime(total);
}

void Node::Clock(int line_num)
{
    boost::recursive_mutex::scoped_lock lock(pImpl_->mtx);
    pImpl_->timings.push_back(std::make_pair(clock(), line_num));
}


void Node::reset()
{
    
    Init(false);
}




Node::Ptr Node::AddChild(Node* child)
{
    return AddChild(Node::Ptr(child));
}
Node::Ptr Node::AddChild(Node::Ptr child)
{
    
    if (child == nullptr)
        return child;
    if(std::find(children.begin(), children.end(), child) != children.end())
        return child;
    int count = 0;
    for(size_t i = 0; i < children.size(); ++i)
    {
        if(children[i]->GetName() == child->GetName())
            ++count;
    }
    
    child->SetDataStream(GetDataStream());
    child->AddParent(this);
    std::string node_name = child->GetTypeName();
    //child->setTreeName(node_name + "-" + boost::lexical_cast<std::string>(count));
    child->Init(true);
    children.push_back(child);
    LOG(trace) << "[ " << GetTreeName() << " ]" << " Adding child " << child->treeName;
    return child;
}

Node::Ptr Node::GetChild(const std::string& treeName)
{
    for(size_t i = 0; i < children.size(); ++i)
    {
        if(children[i]->treeName == treeName)
            return children[i];
    }
    for(size_t i = 0; i < children.size(); ++i)
    {
        if(children[i]->GetName() == treeName)
            return children[i];
    }
    return Node::Ptr();
}


Node::Ptr Node::GetChild(const int& index)
{
    
    return children[index];
}
void Node::SwapChildren(int idx1, int idx2)
{
    
    std::iter_swap(children.begin() + idx1, children.begin() + idx2);
}

void Node::SwapChildren(const std::string& name1, const std::string& name2)
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
void Node::SwapChildren(Node::Ptr child1, Node::Ptr child2)
{
    
    auto itr1 = std::find(children.begin(),children.end(), child1);
    if(itr1 == children.end())
        return;
    auto itr2 = std::find(children.begin(), children.end(), child2);
    if(itr2 == children.end())
        return;
    std::iter_swap(itr1,itr2);
}

std::vector<Node*> Node::GetNodesInScope()
{
    
    std::vector<Node*> nodes;
    if(_parents.size())
        _parents[0]->GetNodesInScope(nodes);
    return nodes;
}
Node* Node::GetNodeInScope(const std::string& name)
{
    
    // Check if this is a child node of mine, if not go up
    auto fullTreeName = GetTreeName();
    int ret = name.compare(0, fullTreeName.length(), fullTreeName);
    if(ret == 0)
    {
        // name is a child of current node, or is the current node
        if(fullTreeName.size() == name.size())
            return this;
        std::string childName = name.substr(fullTreeName.size() + 1);
        auto child = GetChild(childName);
        if(child != nullptr)
            return child.Get();
    }
    if(_parents.size())
        return _parents[0]->GetNodeInScope(name);
    return nullptr;
}

void Node::GetNodesInScope(std::vector<Node*> &nodes)
{
    // Perhaps not thread safe?
    
    // First travel to the root node

    if(nodes.size() == 0)
    {
        Node* node = this;
        while(node->_parents.size())
        {
            node = node->_parents[0].Get();
        }
        nodes.push_back(node);
        node->GetNodesInScope(nodes);
        return;
    }
    nodes.push_back(this);
    for(size_t i = 0; i < children.size(); ++i)
    {
        if(children[i] != nullptr)
            children[i]->GetNodesInScope(nodes);
    }
}


/*Node* Node::GetChildRecursive(std::string treeName_)
{
    

    // TODO tree structure parsing and correct directing of the search
    // Find the common base between this node and treeName


    return nullptr;
}*/

void Node::RemoveChild(Node::Ptr node)
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
void Node::RemoveChild(int idx)
{
    children.erase(children.begin() + idx);
}

void Node::RemoveChild(const std::string &name)
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

void Node::RemoveChild(Node* node)
{
    auto itr = std::find(children.begin(), children.end(), node);
    if(itr != children.end())
    {
        children.erase(itr);
    }
}

void Node::RemoveChild(rcc::weak_ptr<Node> node)
{
    auto itr = std::find(children.begin(), children.end(), node.Get());
    if(itr != children.end())
    {
        children.erase(itr);
    }
}
/*cv::cuda::GpuMat Node::process(cv::cuda::GpuMat &img, cv::cuda::Stream& stream)
{
    if(boost::this_thread::interruption_requested())
        return img;
    //ui_collector::setNode(this);
    ui_collector::set_node_name(getFullTreeName());
    
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
                std::lock_guard<std::recursive_mutex> lock(_mtx);
                auto allocator = dynamic_cast<PitchedAllocator*>(cv::cuda::GpuMat::defaultAllocator());
                if(allocator)
                {
                    allocator->SetScope(this->getTreeName());
                }

                // Do I lock each parameters mutex or do I just lock each node?
                // I should only lock each node, but then I need to make sure the UI keeps track of the node
                // to access the node's mutex while accessing a parameter, for now this works though.
                std::vector<boost::recursive_mutex::scoped_lock> locks;
                for (size_t i = 0; i < _parameters.size(); ++i)
                {
                    locks.push_back(boost::recursive_mutex::scoped_lock(_parameters[i]->mtx));
                }
                TIME
                _rmt_BeginCPUSample(fullTreeName.c_str(), &rmt_hash);
                _rmt_BeginCUDASample(fullTreeName.c_str(), &rmt_cuda_hash, cv::cuda::StreamAccessor::getStream(stream));
                PROFILE_OBJ(fullTreeName.c_str());
                img = doProcess(img, stream);
                rmt_EndCPUSample();
                rmt_EndCUDASample(cv::cuda::StreamAccessor::getStream(stream));
                EndProcessingTime();
            }
            NODE_LOG(trace) << "End:   " << fullTreeName;
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
            std::lock_guard<std::recursive_mutex> lock(_mtx);
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
                ui_collector::set_node_name(getFullTreeName());
                NODE_LOG(error) << "Null child with idx: " + boost::lexical_cast<std::string>(i);
            }
        }
        ui_collector::set_node_name(getFullTreeName());
        // So here is the debate of is a node's output the output of it, or the output of its children....
        // img = childResults;
    }CATCH_MACRO;
    ui_collector::set_node_name("");
    
    return img;
}
TS<SyncedMemory> Node::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    TS < SyncedMemory> output = input;
    if(pre_check(input))
    {
        _current_timestamp = input.frame_number;
        if (boost::this_thread::interruption_requested())
            return output;
        ui_collector::set_node_name(getFullTreeName());
        try
        {
                ClearProcessingTime();
                std::lock_guard<std::recursive_mutex> lock(_mtx);
                auto allocator = dynamic_cast<PitchedAllocator*>(cv::cuda::GpuMat::defaultAllocator());
                if (allocator)
                {
                    allocator->SetScope(this->getTreeName());
                }

                // Do I lock each parameters mutex or do I just lock each node?
                // I should only lock each node, but then I need to make sure the UI keeps track of the node
                // to access the node's mutex while accessing a parameter, for now this works though.
                
                TIME
                _rmt_BeginCPUSample(fullTreeName.c_str(), &rmt_hash);
                _rmt_BeginCUDASample(fullTreeName.c_str(), &rmt_cuda_hash, cv::cuda::StreamAccessor::getStream(stream));
                PROFILE_OBJ(fullTreeName.c_str());
                output = doProcess(input, stream);
                rmt_EndCPUSample();
                rmt_EndCUDASample(cv::cuda::StreamAccessor::getStream(stream));
                EndProcessingTime();
        }CATCH_MACRO
        try
        {
            if (children.size() == 0)
                return output;
            std::vector<Node::Ptr>  children_;
            {
                // Prevents adding of children while running, debatable how much this is needed
                std::lock_guard<std::recursive_mutex> lock(_mtx);
                children_ = children;
            }
            for (size_t i = 0; i < children_.size(); ++i)
            {
                if (children_[i] != nullptr)
                {
                    try
                    {
                        children_[i]->process(output, stream);
                    }CATCH_MACRO
                }
                else
                {
                    ui_collector::set_node_name(getFullTreeName());
                    NODE_LOG(error) << "Null child with idx: " + boost::lexical_cast<std::string>(i);
                }
            }
            ui_collector::set_node_name(getFullTreeName());
        }CATCH_MACRO;
        ui_collector::set_node_name("");
    }
    return output;
}


bool Node::pre_check(const TS<SyncedMemory>& input)
{
    return !input.empty() && enabled;
}
*/

void Node::SetDataStream(IDataStream* stream_)
{
    if (_dataStream)
    {
        LOG(debug) << "Updating stream manager to a new manager";
    }    
    else
    {
        LOG(debug) << "Setting stream manager";
    }
    _dataStream = stream_;
    SetupSignals(_dataStream->GetRelayManager());
    SetupVariableManager(_dataStream->GetVariableManager().get());
    //pImpl_->update_signal = stream_->GetRelayManager()->get_signal<void(Node*)>("NodeUpdated");
    for (auto& child : children)
    {
        child->SetDataStream(_dataStream);
    }
}

IDataStream* Node::GetDataStream()
{
    if (_parents.size() && _dataStream == nullptr)
    {
        LOG(debug) << "Setting data stream from parent";
        SetDataStream(_parents[0]->GetDataStream());
    }
    if (_parents.size() == 0 && _dataStream == nullptr)
    {
        
    }    
    return _dataStream;
}
/*cv::cuda::GpuMat
Node::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream )
{
    return img;
}
TS<SyncedMemory> Node::doProcess(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    TS<SyncedMemory> output = input;
    output.GetGpuMatMutable(stream) = doProcess(input.GetGpuMatMutable(stream), stream);
    return output;
}*/




std::string Node::GetName() const
{
    return treeName;
}

std::string Node::GetTreeName() const
{
    return treeRoot + ":" + treeName;
}


/*std::string Node::GetTreeName()
{
    if(!treeName.size())
    {
        treeName = getName();
        fullTreeName = getName();
    }
        
    return treeName;
}*/
/*std::string Node::GetTreeRoot()
{
    if(!fullTreeName.size())
        fullTreeName = getName();
    return fullTreeName;
}*/

std::vector<rcc::weak_ptr<Node>> Node::GetParents()
{
    return _parents;
}


void Node::Init(bool firstInit)
{
    //ui_collector::set_node_name(getFullTreeName());
    // Node init should be called first because it is where implicit parameters should be setup
    // Then in ParmaeteredIObject, the implicit parameters will be added back to the _parameter vector
    
    NodeInit(firstInit); 
    IMetaObject::Init(firstInit);
}

void Node::NodeInit(bool firstInit)
{

}

void Node::Init(const std::string &configFile)
{
    //ui_collector::set_node_name(getFullTreeName());
    
}


void Node::Init(const cv::FileNode& configNode)
{
    //ui_collector::set_node_name(getFullTreeName());
    LOG(trace) << " Initializing from file";
    //configNode["NodeName"] >> nodeName;
    configNode["NodeTreeName"] >> treeName;
    configNode["TreeRoot"] >> treeRoot;
    configNode["Enabled"] >> enabled;
    configNode["ExternalDisplay"] >> externalDisplay;
    cv::FileNode childrenFS = configNode["Children"];
    int childCount = (int)childrenFS["Count"];
    for(int i = 0; i < childCount; ++i)
    {
        cv::FileNode childNode = childrenFS["Node-" + boost::lexical_cast<std::string>(i)];
        std::string name = (std::string)childNode["NodeName"];
        auto node = NodeFactory::Instance()->AddNode(name);
        if (node != nullptr)
        {
            AddChild(node);
            node->Init(childNode);
            //ui_collector::set_node_name(getFullTreeName());
        }
        else
        {
            LOG(error) << "No node found with the name " << name;
        }
    }
    cv::FileNode paramNode = configNode["Parameters"];
    // #TODO proper serialization with cereal
    /*for (size_t i = 0; i < _parameters.size(); ++i)
    {
        try
        {
            if (_parameters[i]->type & Parameters::Parameter::Input)
            {
                auto node = paramNode[_parameters[i]->GetName()];
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
                            auto inputParam = dynamic_cast<mo::InputParameter*>(_parameters[i]);
                            inputParam->SetInput(param);
                        }
                    }
                }

            }
            else
            {
                // #TODO update to new api
                if (_parameters[i]->CheckFlags(mo::Control_e))
                    Parameters::Persistence::cv::DeSerialize(&paramNode, _parameters[i]);
            }
        }
        catch (cv::Exception &e)
        {
            LOG(error) << "Deserialization failed for " << _parameters[i]->GetName() << " with type " << _parameters[i]->GetTypeInfo().name() << std::endl;
        }
    }*/
}

void Node::Serialize(ISimpleSerializer *pSerializer)
{
    LOG(trace) << " Serializing";
    IMetaObject::Serialize(pSerializer);
    SERIALIZE(children);
    SERIALIZE(treeName);
    SERIALIZE(treeRoot);
    SERIALIZE(_parents);
    SERIALIZE(externalDisplay);
    SERIALIZE(enabled);
    SERIALIZE(pImpl_);
    SERIALIZE(_dataStream);
}

void
Node::Serialize(cv::FileStorage& fs)
{
    /*NODE_LOG(trace) << " Serializing to file";
    if(fs.isOpened())
    {
        fs << "NodeName" << GetTypeName();
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
        for(size_t i = 0; i < _parameters.size(); ++i)
        {
            if (_parameters[i]->type & Parameters::Parameter::Input)
            {
                auto inputParam = dynamic_cast<Parameters::InputParameter*>(_parameters[i]);
                if (inputParam)
                {
                    auto input = inputParam->GetInput();
                    if (input)
                    {
                        fs << _parameters[i]->GetName().c_str() << "{";
                        fs << "TreeName" << _parameters[i]->GetTreeName();
                        fs << "InputParameter" << input->GetTreeName();
                        fs << "Type" << _parameters[i]->GetTypeInfo().name();
                        auto toolTip = _parameters[i]->GetTooltip();
                        if (toolTip.size())
                            fs << "ToolTip" << toolTip;
                        fs << "}";
                    }
                }
            }
            else
            {
                if (_parameters[i]->type & Parameters::Parameter::Control)
                {
                    // TODO
                    try
                    {
                        Parameters::Persistence::cv::Serialize(&fs, _parameters[i]);
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
    */
}

void Node::SetTreeName(const std::string& name)
{
    treeName = name;
    /*std::string fullTreeName_;
    if (parent == nullptr)
        fullTreeName_ = treeName;
    else
        fullTreeName_ = parent->getFullTreeName() + "." + treeName;
    setFullTreeName(fullTreeName_);
    for(size_t i = 0; i < children.size(); ++i)
    {
        children[i]->setTreeName(children[i]->treeName);
    }*/
}
/*void Node::setFullTreeName(const std::string& name)
{
    SetParameterRoot(name);
    //fullTreeName = name;
}*/

void Node::AddParent(Node* parent_)
{
    _parents.push_back(parent_);
}




