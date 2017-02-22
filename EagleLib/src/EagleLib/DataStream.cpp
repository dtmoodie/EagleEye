#include "EagleLib/IDataStream.hpp"
#include "EagleLib/DataStream.hpp"
#include "EagleLib/rcc/SystemTable.hpp"
#include "EagleLib/utilities/sorting.hpp"
#include "EagleLib/Logging.h"


#include "EagleLib/ParameterBuffer.h"
#include "EagleLib/IVariableSink.h"
#include "EagleLib/IViewManager.h"
#include "EagleLib/ICoordinateManager.h"
#include "EagleLib/rendering/RenderingEngine.h"
#include "EagleLib/tracking/ITrackManager.h"
#include "EagleLib/Nodes/IFrameGrabber.hpp"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/NodeFactory.h"

#include "MetaObject/Signals/TypedSlot.hpp"
#include "MetaObject/Signals/RelayManager.hpp"
#include "MetaObject/Parameters/VariableManager.h"
#include "MetaObject/Thread/InterThread.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include <MetaObject/Logging/Profiling.hpp>
#include "MetaObject/IO/memory.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"

#include <opencv2/core.hpp>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <fstream>
#include <future>

using namespace EagleLib;
using namespace EagleLib::Nodes;
INSTANTIATE_META_PARAMETER(rcc::shared_ptr<IDataStream>);
INSTANTIATE_META_PARAMETER(rcc::weak_ptr<IDataStream>);
#define CATCH_MACRO                                                         \
    catch (boost::thread_resource_error& err)                               \
{                                                                           \
    LOG(error) << err.what();                                          \
}                                                                           \
catch (boost::thread_interrupted& err)                                      \
{                                                                           \
    LOG(error) << "Thread interrupted";                                \
    /* Needs to pass this back up to the chain to the processing thread.*/  \
    /* That way it knowns it needs to exit this thread */                   \
    throw err;                                                              \
}                                                                           \
catch (boost::thread_exception& err)                                        \
{                                                                           \
    LOG(error) << err.what();                                          \
}                                                                           \
    catch (cv::Exception &err)                                              \
{                                                                           \
    LOG(error) << err.what();                                          \
}                                                                           \
    catch (boost::exception &err)                                           \
{                                                                           \
    LOG(error) << "Boost error";                                       \
}                                                                           \
catch (std::exception &err)                                                 \
{                                                                           \
    LOG(error) << err.what();                                            \
}                                                                           \
catch (...)                                                                 \
{                                                                           \
    LOG(error) << "Unknown exception";                                 \
}



 


// **********************************************************************
//              DataStream
// **********************************************************************
DataStream::DataStream()
{
    _sig_manager = GetRelayManager();
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        mo::RelayManager* global_signal_manager = table->GetSingleton<mo::RelayManager>();
        if (!global_signal_manager)
        {
            global_signal_manager  = mo::RelayManager::Instance();
            table->SetSingleton<mo::RelayManager>(global_signal_manager);
        }
        global_signal_manager->ConnectSlots(this);
        global_signal_manager->ConnectSignals(this);
    }
    GetRelayManager()->ConnectSlots(this);
    GetRelayManager()->ConnectSignals(this);
    stream_id = 0;
    _thread_id = 0;
    _processing_thread = mo::ThreadPool::Instance()->RequestThread();
    _processing_thread.SetInnerLoop(GetSlot_process<int(void)>());
    _processing_thread.SetThreadName("DataStreamThread");
    this->_ctx = this->_processing_thread.GetContext();
}

void DataStream::node_updated(Nodes::Node* node)
{
    dirty_flag = true;
}

void DataStream::update()
{
    dirty_flag = true;
}

void DataStream::parameter_updated(mo::IMetaObject* obj, mo::IParameter* param)
{
    dirty_flag = true;
}

void DataStream::parameter_added(mo::IMetaObject* obj, mo::IParameter* param)
{
    dirty_flag = true;
}

void DataStream::run_continuously(bool value)
{
    
}

void DataStream::InitCustom(bool firstInit)
{
    if(firstInit)
    {
        this->SetupSignals(GetRelayManager());
    }
    _processing_thread.Start();
}

DataStream::~DataStream()
{
    StopThread();
    top_level_nodes.clear();
    relay_manager.reset();
    _sig_manager = nullptr;
    for(auto thread : connection_threads)
    {
        thread->join();
        delete thread;
    }
}

mo::Context* DataStream::GetContext() const
{
    return IDataStream::GetContext();
}

std::vector<rcc::weak_ptr<EagleLib::Nodes::Node>> DataStream::GetTopLevelNodes()
{
    std::vector<rcc::weak_ptr<EagleLib::Nodes::Node>> output;
    for(auto& itr : top_level_nodes)
    {
        output.emplace_back(itr);
    }
    return output;
}
rcc::weak_ptr<IViewManager> DataStream::GetViewManager()
{
    return view_manager;
}

// Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
rcc::weak_ptr<ICoordinateManager> DataStream::GetCoordinateManager()
{
    return coordinate_manager;
}

// Handles actual rendering of data.  Use for adding extra objects to the scene
rcc::weak_ptr<IRenderEngine> DataStream::GetRenderingEngine()
{
    return rendering_engine;
}

// Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
rcc::weak_ptr<ITrackManager> DataStream::GetTrackManager()
{
    return track_manager;
}

mo::RelayManager* DataStream::GetRelayManager()
{
    if (relay_manager == nullptr)
        relay_manager.reset(new mo::RelayManager());
    return relay_manager.get();
}

IParameterBuffer* DataStream::GetParameterBuffer()
{
    if (_parameter_buffer == nullptr)
        _parameter_buffer.reset(new ParameterBuffer(10));
    return _parameter_buffer.get();

}
rcc::weak_ptr<WindowCallbackHandler> DataStream::GetWindowCallbackManager()
{
    if(!_window_callback_handler)
    {
        _window_callback_handler = WindowCallbackHandler::Create();
        _window_callback_handler->SetupSignals(this->GetRelayManager());
    }
    return _window_callback_handler;
}

std::shared_ptr<mo::IVariableManager> DataStream::GetVariableManager()
{
    if(variable_manager == nullptr)
        variable_manager.reset(new mo::VariableManager());
    return variable_manager;
}

bool DataStream::LoadDocument(const std::string& document, const std::string& prefered_loader)
{
    std::string file_to_load = document;
    if(file_to_load.size() == 0)
        return false;
    if(file_to_load.at(0) == '\"' && file_to_load.at(file_to_load.size() - 1) == '\"')
    {
        file_to_load = file_to_load.substr(1, file_to_load.size() - 2);
    }
    std::lock_guard<std::mutex> lock(nodes_mtx);

    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(EagleLib::Nodes::IFrameGrabber::s_interfaceID);
    std::vector<IObjectConstructor*> valid_frame_grabbers;
    std::vector<int> frame_grabber_priorities;
    if(constructors.empty())
    {
        LOG(warning) << "No frame grabbers found";
        return false;
    }
    for(auto& constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(info)
        {
            auto fg_info = dynamic_cast<Nodes::FrameGrabberInfo*>(info);
            if(fg_info)
            {
                int priority = fg_info->CanLoadDocument(file_to_load);
                if(priority != 0)
                {
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(priority);
                }
            }
        }
    }

    if(valid_frame_grabbers.empty())
    {
        auto f = [&constructors]()->std::string
        {
            std::stringstream ss;
            for(auto& constructor : constructors)
            {
                ss << constructor->GetName() << ", ";
            }
            return ss.str();
        };
        LOG(warning) << "No valid frame grabbers for " << file_to_load
                     << " framegrabbers: " << f();

        return false;
    }
    // Pick the frame grabber with highest priority
    
    auto idx = sort_index_descending(frame_grabber_priorities);
    if(prefered_loader.size())
    {
        for(int i = 0; i < valid_frame_grabbers.size(); ++i)
        {
            if(prefered_loader == valid_frame_grabbers[i]->GetName())
            {
                idx.insert(idx.begin(), i);
                break;
            }
        }
    }
    
    for(int i = 0; i < idx.size(); ++i)
    {
        auto fg = rcc::shared_ptr<IFrameGrabber>(valid_frame_grabbers[idx[i]]->Construct());
        auto fg_info = dynamic_cast<FrameGrabberInfo*>(valid_frame_grabbers[idx[i]]->GetObjectInfo());
        fg->InitializeFrameGrabber(this);
        fg->Init(true);
        //std::promise<bool> promise;
        struct thread_load_object
        {
            std::promise<bool> promise;
            rcc::shared_ptr<IFrameGrabber> fg;
            std::string document;
            void load()
            {
                promise.set_value(fg->LoadFile(document));
            }
        };
        auto obj = new thread_load_object();
        obj->fg = fg;
        obj->document = file_to_load;
        auto future = obj->promise.get_future();
        boost::thread* connection_thread = new boost::thread([obj]()->void{
            try
            {
                obj->load();
            }catch(cv::Exception&e)
            {
                LOG(debug) << e.what();
            }
            
            delete obj;
        });
        if(connection_thread->timed_join(boost::posix_time::milliseconds(fg_info->LoadTimeout())))
        {
            if(future.get())
            {
                top_level_nodes.emplace_back(fg);
                LOG(info) << "Loading " << file_to_load << " with frame_grabber: " << fg->GetTypeName() << " with priority: " << frame_grabber_priorities[idx[i]];
                delete connection_thread;
                return true; // successful load
            }else // unsuccessful load
            {
                LOG(warning) << "Unable to load " << file_to_load << " with " << fg_info->GetObjectName();
            }
        }
        else // timeout        
        {
            LOG(warning) << "Timeout while loading " << file_to_load << " with " << fg_info->GetObjectName() << " after waiting " << fg_info->LoadTimeout() << " ms";
            connection_threads.push_back(connection_thread);
        }
    }
    return false;
}
bool IDataStream::CanLoadDocument(const std::string& document)
{
    std::string doc_to_load = document;
    if(doc_to_load.size() == 0)
        return false;
    if(doc_to_load.at(0) == '\"' && doc_to_load.at(doc_to_load.size() - 1) == '\"')
    {
        doc_to_load = doc_to_load.substr(1, doc_to_load.size() - 2);
    }

    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(EagleLib::Nodes::IFrameGrabber::s_interfaceID);
    std::vector<IObjectConstructor*> valid_frame_grabbers;
    std::vector<int> frame_grabber_priorities;
    if (constructors.empty())
    {
        LOG(warning) << "No frame grabbers found";
        return false;
    }
    for (auto& constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if (info)
        {
            auto fg_info = dynamic_cast<FrameGrabberInfo*>(info);
            if (fg_info)
            {
                int priority = fg_info->CanLoadDocument(doc_to_load);
                if (priority != 0)
                {
                    LOG(debug) << fg_info->GetObjectName() << " can load document";
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(priority);
                }
            }
        }
    }
    return !valid_frame_grabbers.empty();
}

std::vector<rcc::shared_ptr<Nodes::Node>> DataStream::GetNodes() const
{
    return top_level_nodes;
}
std::vector<rcc::shared_ptr<Nodes::Node>> DataStream::GetAllNodes() const
{
    std::vector<rcc::shared_ptr<Nodes::Node>> output;
    for(auto& child : child_nodes)
    {
        output.emplace_back(child);
    }
    return output;
}
std::vector<rcc::shared_ptr<Nodes::Node>> DataStream::AddNode(const std::string& nodeName)
{
    return EagleLib::NodeFactory::Instance()->AddNode(nodeName, this);
}
void DataStream::AddNode(rcc::shared_ptr<Nodes::Node> node)
{
    node->SetDataStream(this);
    if(!_processing_thread.IsOnThread() && _processing_thread.GetIsRunning())
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();

        _processing_thread.PushEventQueue(std::bind([&promise, node, this]()
        {
            rcc::shared_ptr<Node> node_ = node;
            if (std::find(top_level_nodes.begin(), top_level_nodes.end(), node) != top_level_nodes.end())
                return;
            if (node->name.size() == 0)
            {
                std::string node_name = node->GetTypeName();
                int count = 0;
                for (size_t i = 0; i < top_level_nodes.size(); ++i)
                {
                    if (top_level_nodes[i] && top_level_nodes[i]->GetTypeName() == node_name)
                        ++count;
                }
                node_->SetUniqueId(count);
            }
            node_->SetParameterRoot(node_->GetTreeName());
            top_level_nodes.push_back(node);
            dirty_flag = true;
            promise.set_value();
        }));
        future.wait();
        return;
    }
    if(std::find(top_level_nodes.begin(), top_level_nodes.end(), node) != top_level_nodes.end())
        return;
    if(node->name.size()  == 0)
    {
        std::string node_name = node->GetTypeName();
        int count = 0;
        for (size_t i = 0; i < top_level_nodes.size(); ++i)
        {
            if (top_level_nodes[i] && top_level_nodes[i]->GetTypeName() == node_name)
                ++count;
        }
        node->SetUniqueId(count);
    }
    node->SetParameterRoot(node->GetTreeName());
    top_level_nodes.push_back(node);
    dirty_flag = true;
}
void DataStream::AddChildNode(rcc::shared_ptr<Nodes::Node> node)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    if(std::find(child_nodes.begin(), child_nodes.end(), node.Get()) != child_nodes.end())
        return;
    int type_count = 0;
    for(auto& child : child_nodes)
    {
        if(child && child != node && child->GetTypeName() == node->GetTypeName())
            ++type_count;
    }
    node->SetUniqueId(type_count);
    child_nodes.emplace_back(node);
}
void DataStream::RemoveChildNode(rcc::shared_ptr<Nodes::Node> node)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    std::remove(child_nodes.begin(), child_nodes.end(), node);
}
void DataStream::AddNodeNoInit(rcc::shared_ptr<Nodes::Node> node)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    top_level_nodes.push_back(node);
    dirty_flag = true;
}
void DataStream::AddNodes(std::vector<rcc::shared_ptr<Nodes::Node>> nodes)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    for (auto& node : nodes)
    {
        node->SetDataStream(this);
    }
    if(!_processing_thread.IsOnThread())
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        _processing_thread.PushEventQueue(std::bind([&nodes, this, &promise]()
        {
            for (auto& node : nodes)
            {
                AddNode(node);
            }
            dirty_flag = true;
            promise.set_value();
        }));
        future.wait();
    }
    for (auto& node : nodes)
    {
        top_level_nodes.push_back(node);
    }
    dirty_flag = true;
}
void DataStream::RemoveNode(Nodes::Node* node)
{
    {
        std::lock_guard<std::mutex> lock(nodes_mtx);
        std::remove(top_level_nodes.begin(), top_level_nodes.end(), node);
    }
    
    RemoveChildNode(node);
}

void DataStream::RemoveNode(rcc::shared_ptr<Nodes::Node> node)
{
    {
        std::lock_guard<std::mutex> lock(nodes_mtx);
        std::remove(top_level_nodes.begin(), top_level_nodes.end(), node);
    }
    RemoveChildNode(node);
}

Nodes::Node* DataStream::GetNode(const std::string& nodeName)
{
    
    std::lock_guard<std::mutex> lock(nodes_mtx);
    for(auto& node : top_level_nodes)
    {
        if(node) // during serialization top_level_nodes is resized thus allowing for nullptr nodes until they are serialized
        {
            auto found_node = node->GetNodeInScope(nodeName);
            if(found_node)
            {
                return found_node;
            }
        }       
    }
    
    return nullptr;
}

void DataStream::AddVariableSink(IVariableSink* sink)
{
    variable_sinks.push_back(sink);
}

void DataStream::RemoveVariableSink(IVariableSink* sink)
{
    std::remove_if(variable_sinks.begin(), variable_sinks.end(), [sink](IVariableSink* other)->bool{return other == sink;});
}
void DataStream::StartThread()
{
    StopThread();
    sig_StartThreads();
    _processing_thread.Start();
}

void DataStream::StopThread()
{
    _processing_thread.Stop();
    sig_StopThreads();
}


void DataStream::PauseThread()
{
    sig_StopThreads();
    _processing_thread.Stop();
}

void DataStream::ResumeThread()
{
    _processing_thread.Start();
    sig_StartThreads();
}

int DataStream::process()
{
    if (dirty_flag/* || run_continuously == true*/)
    {
        dirty_flag = false;
        //mo::scoped_profile profile("Processing nodes", &rmt_hash, &rmt_cuda_hash, &_context.GetStream());
        mo::scoped_profile profile_nodes("Processing nodes", &_rmt_hash, &_rmt_cuda_hash, &GetContext()->GetStream());
        for (auto& node : top_level_nodes)
        {
            node->Process();
        }
        if (dirty_flag)
        {
            return 10;
        }
    }
    return 10;
}

IDataStream::Ptr IDataStream::Create(const std::string& document, const std::string& preferred_frame_grabber)
{
    //auto stream = mo::MetaObjectFactory::Instance()->Create<IDataStream>("DataStream");
    auto stream = DataStream::Create();
    auto fg = IFrameGrabber::Create(document, preferred_frame_grabber);
    if(fg)
    {
        stream->AddNode(fg);
        return stream;
    }
    return IDataStream::Ptr();
}
std::unique_ptr<ISingleton>& DataStream::GetSingleton(mo::TypeInfo type)
{
    return _singletons[type];
}
std::unique_ptr<ISingleton>& DataStream::GetIObjectSingleton(mo::TypeInfo type)
{
    return _iobject_singletons[type];
}

MO_REGISTER_OBJECT(DataStream)
