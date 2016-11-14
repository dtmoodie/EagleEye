#include "MetaObject/Parameters/MetaParameter.hpp"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/NodeFactory.h"
#include "EagleLib/Nodes/NodeInfo.hpp"
#include "EagleLib/IDataStream.hpp"
#include <EagleLib/rcc/external_includes/cv_videoio.hpp>
#include <EagleLib/rcc/SystemTable.hpp>
#include <EagleLib/utilities/GpuMatAllocators.h>
#include "EagleLib/Signals.h"
#include "EagleLib/Detail/AlgorithmImpl.hpp"
#include <EagleLib/IO/memory.hpp>

#include "../RuntimeObjectSystem/ISimpleSerializer.h"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"


#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Logging/Profiling.hpp>

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

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thrust/system/system_error.h>


#include <regex>
using namespace EagleLib;
using namespace EagleLib::Nodes;
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE



#define CATCH_MACRO                                                         \
    catch(mo::ExceptionWithCallStack<cv::Exception>& e)                     \
{                                                                           \
    LOG_NODE(error) << e.what() << "\n" << e.CallStack();                   \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
    catch(thrust::system_error& e)                                          \
{                                                                           \
    LOG_NODE(error) << e.what();                                            \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
    catch(mo::ExceptionWithCallStack<std::string>& e)                       \
{                                                                           \
    LOG_NODE(error) << std::string(e) << "\n" << e.CallStack();             \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
catch(mo::IExceptionWithCallStackBase& e)                                   \
{                                                                           \
    LOG_NODE(error) << "Exception thrown with callstack: \n" << e.CallStack(); \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
catch (boost::thread_resource_error& err)                                   \
{                                                                           \
    LOG_NODE(error) << err.what();                                          \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
catch (boost::thread_interrupted& err)                                      \
{                                                                           \
    LOG_NODE(error) << "Thread interrupted";                                \
    /* Needs to pass this back up to the chain to the processing thread.*/  \
    /* That way it knowns it needs to exit this thread */                   \
    throw err;                                                              \
}                                                                           \
catch (boost::thread_exception& err)                                        \
{                                                                           \
    LOG_NODE(error) << err.what();                                          \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
    catch (cv::Exception &err)                                              \
{                                                                           \
    LOG_NODE(error) << err.what();                                          \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
    catch (boost::exception &err)                                           \
{                                                                           \
    LOG_NODE(error) << "Boost error";                                       \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
catch (std::exception &err)                                                 \
{                                                                           \
    LOG_NODE(error) << err.what();                                          \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}                                                                           \
catch (...)                                                                 \
{                                                                           \
    LOG_NODE(error) << "Unknown exception";                                 \
    ++_pimpl_node->throw_count;                                             \
    if(_pimpl_node->throw_count > 100)                                      \
        _pimpl_node->disable_due_to_errors = true;                          \
}


std::string NodeInfo::Print() const
{
    return mo::IMetaObjectInfo::Print();
}


namespace EagleLib
{
    namespace Nodes
    {
        class NodeImpl
        {
        public:
            long long throw_count = 0;
            bool disable_due_to_errors = false;
#ifdef _DEBUG
        std::vector<long long> timestamps;
#endif
        };
    }    
}

Node::Node()
{
    _modified = true;
    _pimpl_node.reset(new NodeImpl());
}

bool Node::ConnectInput(rcc::shared_ptr<Node> node, const std::string& output_name, const std::string& input_name, mo::ParameterTypeFlags type)
{
    auto output = node->GetOutput(output_name);
    auto input = this->GetInput(input_name);
    if(output && input)
    {
        if(this->IMetaObject::ConnectInput(input, node.Get(), output, type))
        {
            AddParent(node.Get());
            return true;
        }else
        {
            return false;
        }
    }
    if(output == nullptr)
    {
        auto outputs = node->GetOutputs();
        auto f = [&outputs]() ->std::string
        {
            std::stringstream ss;
            for(auto& output : outputs)
            {
                ss << output->GetName() << " ";
            }
            return ss.str();
        };
        LOG(debug) << "Unable to find output with name \"" << output_name << "\" in node \"" << node->GetTreeName() << "\".  Existing outputs: " << f();;
    }
    if(input == nullptr)
    {
        auto outputs = node->GetInputs();
        auto f = [&outputs]()->std::string
        {
            std::stringstream ss;
            for(auto& output : outputs)
            {
                ss << output->GetName() << " ";
            }
            return ss.str();
        };
        LOG(debug) << "Unable to find input with name \"" << input_name << "\" in node \"" << this->GetTreeName() << "\". Existing inputs: " << f();
    }
    return false;
}
bool Node::ConnectInput(rcc::shared_ptr<Node> output_node,    mo::IParameter* output_param,    mo::InputParameter* input_param,    mo::ParameterTypeFlags type)
{
    if (this->IMetaObject::ConnectInput(input_param, output_node.Get(), output_param, type))
    {
        AddParent(output_node.Get());
        return true;
    }
    else
    {
        return false;
    }
}

bool Node::CheckInputs()
{
    if(_pimpl->_sync_method == Algorithm::SyncEvery && _pimpl->_ts_processing_queue.size() != 0)
        _modified = true;
    if(_modified == false)
    {
        LOG(trace) << "_modified == false for " << GetTreeName();
        return false;
    }
    
    return Algorithm::CheckInputs();
}

void Node::onParameterUpdate(mo::Context* ctx, mo::IParameter* param)
{
    Algorithm::onParameterUpdate(ctx, param);
    if(param->CheckFlags(mo::Control_e) || param->CheckFlags(mo::Input_e))
    {
        _modified = true;
        // Reanble if disabled 
        if(_pimpl_node->disable_due_to_errors)
        {
            _pimpl_node->throw_count--;
            _pimpl_node->disable_due_to_errors = false;
        }
    }
}

bool Node::Process()
{
    if(_enabled == false)
        return false;
    if(_modified == false)
        return false;
    if(_pimpl_node->disable_due_to_errors)
        return false;
    { // scope of the lock
        boost::recursive_mutex::scoped_lock lock(*_mtx);

        if (!CheckInputs())
        {
            return false;
        }

        _modified = false;
        
        {
            //mo::scoped_profile profiler(this->GetTreeName().c_str(), &this->_rmt_hash, &this->_rmt_cuda_hash, &Stream());
            try
            {
                if (!ProcessImpl())
                    return false;
            }CATCH_MACRO
            
        }
        

        _pimpl->last_ts = _pimpl->ts;
        if (_pimpl->sync_input == nullptr && _pimpl->ts != -1)
            ++_pimpl->ts;
        if (_pimpl->_sync_method == SyncEvery && _pimpl->sync_input)
        {
            boost::recursive_mutex::scoped_lock lock(_pimpl->_mtx);
            if (_pimpl->ts == _pimpl->_ts_processing_queue.front())
            {
                _pimpl->_ts_processing_queue.pop();
            }
        }
    } // end lock
    
    for(rcc::shared_ptr<Node>& child : _children)
    {
        if(child->_ctx && this->_ctx)
        {
            if(child->_ctx->thread_id == this->_ctx->thread_id)
            {
                child->Process();
            }
        }else
        {
            child->Process();
        }
    }
    return true;
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
    if(_ctx && mo::GetThisThread() != _ctx->thread_id)
    {
        mo::ThreadSpecificQueue::Push(std::bind((Node::Ptr(Node::*)(Node::Ptr))&Node::AddChild, this, child), _ctx->thread_id, this);
        return child;
    }
    if (child == nullptr)
        return child;
    if(std::find(_children.begin(), _children.end(), child) != _children.end())
        return child;
    if(child == this) // This can happen based on a bad user config
        return child;
    int count = 0;
    for(size_t i = 0; i < _children.size(); ++i)
    {
        if(_children[i] && _children[i]->GetTypeName() == child->GetTypeName())
            ++count;
    }
    _children.push_back(child);
    child->SetDataStream(GetDataStream());
    child->AddParent(this);
    child->SetContext(this->_ctx, false);
    std::string node_name = child->GetTypeName();
    child->SetUniqueId(count);
    child->SetParameterRoot(child->GetTreeName());
    LOG(trace) << "[ " << GetTreeName() << " ]" << " Adding child " << child->GetTreeName();
    return child;
}

Node::Ptr Node::GetChild(const std::string& treeName)
{
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    for(size_t i = 0; i < _children.size(); ++i)
    {
        if(_children[i]->GetTreeName()== treeName)
            return _children[i];
    }
    for(size_t i = 0; i < _children.size(); ++i)
    {
        if(_children[i]->GetTreeName() == treeName)
            return _children[i];
    }
    return Node::Ptr();
}


Node::Ptr Node::GetChild(const int& index)
{
    return _children[index];
}
std::vector<Node::Ptr>   Node::GetChildren()
{
    return _children;
}

void Node::SwapChildren(int idx1, int idx2)
{
    
    std::iter_swap(_children.begin() + idx1, _children.begin() + idx2);
}

void Node::SwapChildren(const std::string& name1, const std::string& name2)
{
    
    auto itr1 = _children.begin();
    auto itr2 = _children.begin();
    for(; itr1 != _children.begin(); ++itr1)
    {
        if((*itr1)->GetTreeName() == name1)
            break;
    }
    for(; itr2 != _children.begin(); ++itr2)
    {
        if((*itr2)->GetTreeName() == name2)
            break;
    }
    if(itr1 != _children.end() && itr2 != _children.end())
        std::iter_swap(itr1,itr2);
}

void Node::SwapChildren(Node::Ptr child1, Node::Ptr child2)
{
    
    auto itr1 = std::find(_children.begin(),_children.end(), child1);
    if(itr1 == _children.end())
        return;
    auto itr2 = std::find(_children.begin(), _children.end(), child2);
    if(itr2 == _children.end())
        return;
    std::iter_swap(itr1,itr2);
}

std::vector<Node*> Node::GetNodesInScope()
{
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    std::vector<Node*> nodes;
    if(_parents.size())
        _parents[0]->GetNodesInScope(nodes);
    return nodes;
}

Node* Node::GetNodeInScope(const std::string& name)
{
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    if(name == GetTreeName())
        return this;
    for(auto& child : _children)
    {
        auto result = child->GetNodeInScope(name);
        if(result)
        {
            return result;
        }
    }
    return nullptr;
}

void Node::GetNodesInScope(std::vector<Node*> &nodes)
{
    // Perhaps not thread safe?
    
    // First travel to the root node
    boost::recursive_mutex::scoped_lock lock(*_mtx);
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
    for(size_t i = 0; i < _children.size(); ++i)
    {
        if(_children[i] != nullptr)
            _children[i]->GetNodesInScope(nodes);
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
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    for(auto itr = _children.begin(); itr != _children.end(); ++itr)
    {
        if(*itr == node)
        {
            _children.erase(itr);
                    return;
        }
    }
}
void Node::RemoveChild(int idx)
{
    _children.erase(_children.begin() + idx);
}

void Node::RemoveChild(const std::string &name)
{
    for(auto itr = _children.begin(); itr != _children.end(); ++itr)
    {
        if((*itr)->GetTreeName() == name)
        {
            _children.erase(itr);
            return;
        }
    }
}

void Node::RemoveChild(Node* node)
{
    auto itr = std::find(_children.begin(), _children.end(), node);
    if(itr != _children.end())
    {
        _children.erase(itr);
    }
}

void Node::RemoveChild(rcc::weak_ptr<Node> node)
{
    auto itr = std::find(_children.begin(), _children.end(), node.Get());
    if(itr != _children.end())
    {
        _children.erase(itr);
    }
}


void Node::SetDataStream(IDataStream* stream_)
{
    if(stream_ == nullptr)
        return;
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    if (_dataStream && _dataStream != stream_)
    {
        _dataStream->RemoveNode(this);
    }
    _dataStream = stream_;
    this->SetContext(stream_->GetContext());
    SetupSignals(_dataStream->GetRelayManager());
    SetupVariableManager(_dataStream->GetVariableManager().get());
    _dataStream->AddChildNode(this);
    for (auto& child : _children)
    {
        child->SetDataStream(_dataStream.Get());
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
        _dataStream = IDataStream::Create();
        _dataStream->AddNode(this);
    }    
    return _dataStream.Get();
}
std::shared_ptr<mo::IVariableManager>     Node::GetVariableManager()
{
    return GetDataStream()->GetVariableManager();
}

std::string Node::GetTreeName() const
{
    return std::string(GetTypeName()) + boost::lexical_cast<std::string>(_unique_id);
}


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
    /*cv::FileNode paramNode = configNode["Parameters"];
    // #TODO proper serialization with cereal
    for (size_t i = 0; i < _parameters.size(); ++i)
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
    SERIALIZE(_children);
    SERIALIZE(_parents);
    
    SERIALIZE(_pimpl_node);
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


void Node::AddParent(Node* parent_)
{
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    if(std::find(_parents.begin(), _parents.end(), parent_) != _parents.end())
        return;
    _parents.push_back(parent_);
    lock.unlock();
    parent_->AddChild(this);
}

void Node::SetUniqueId(int id)
{
    _unique_id = id;
    SetParameterRoot(GetTreeName());
}