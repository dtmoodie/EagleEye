#include "DataStreamManager.h"
#include <opencv2/core.hpp>
#include "rcc/SystemTable.hpp"
#include "rcc/ObjectManager.h"
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include "utilities/sorting.hpp"
#include "EagleLib/Logging.h"
#include "Remotery.h"
using namespace EagleLib;
// **********************************************************************
//              DataStream
// **********************************************************************
DataStream::DataStream()
{
    //signal_manager.reset(new SignalManager);
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        signal_manager = table->GetSingleton<SignalManager>();
        if(!signal_manager)
        {
            signal_manager = new SignalManager();
            table->SetSingleton<SignalManager>(signal_manager);
            Signals::signal_manager::set_instance(signal_manager);
        }
        connections.push_back(signal_manager->Connect<void(void)>("StopThreads", std::bind(&DataStream::StopProcess, this), this, -1));
        connections.push_back(signal_manager->Connect<void(void)>("StartThreads", std::bind(&DataStream::LaunchProcess, this), this, -1));
    }
    paused = false;
    stream_id = 0;
    
}

DataStream::~DataStream()
{
    StopProcess();

}

int DataStream::get_stream_id()
{
    return stream_id;
}

shared_ptr<IViewManager> DataStream::GetViewManager()
{
    return view_manager;
}

// Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
shared_ptr<ICoordinateManager> DataStream::GetCoordinateManager()
{
    return coordinate_manager;
}

// Handles actual rendering of data.  Use for adding extra objects to the scene
shared_ptr<IRenderEngine> DataStream::GetRenderingEngine()
{
    return rendering_engine;
}

// Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
shared_ptr<ITrackManager> DataStream::GetTrackManager()
{
    return track_manager;
}

// Handles actual loading of the image, etc
shared_ptr<IFrameGrabber> DataStream::GetFrameGrabber()
{
    return frame_grabber;
}

SignalManager* DataStream::GetSignalManager()
{
    return signal_manager;
}

bool DataStream::LoadDocument(const std::string& document)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    auto constructors = ObjectManager::Instance().GetConstructorsForInterface(IID_FrameGrabber);
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
            auto fg_info = dynamic_cast<FrameGrabberInfo*>(info);
            if(fg_info)
            {
                if(fg_info->CanLoadDocument(document))
                {
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(fg_info->Priority());
                }
            }
        }
    }

    if(valid_frame_grabbers.empty())
    {
        LOG(warning) << "No valid frame grabbers for " << document;
        return false;
    }
    // Pick the frame grabber with highest priority
    
    auto idx = sort_index_descending(frame_grabber_priorities);
    for(int i = 0; i < idx.size(); ++i)
    {
        auto fg = shared_ptr<IFrameGrabber>(valid_frame_grabbers[idx[i]]->Construct());
        auto fg_info = static_cast<FrameGrabberInfo*>(valid_frame_grabbers[idx[i]]->GetObjectInfo());
        fg->InitializeFrameGrabber(this);
        //std::promise<bool> promise;
        struct thread_load_object
        {
            std::promise<bool> promise;
            shared_ptr<IFrameGrabber> fg;
            std::string document;
            void load()
            {
                promise.set_value(fg->LoadFile(document));
            }
        };
        auto obj = new thread_load_object();
        obj->fg = fg;
        obj->document = document;
        auto future = obj->promise.get_future();
        boost::thread connection_thread = boost::thread([obj]()->void{
            obj->load();
            delete obj;
        });
        if(connection_thread.timed_join(boost::posix_time::milliseconds(fg_info->LoadTimeout())))
        {
            if(future.get())
            {
                frame_grabber = fg;
                return true; // successful load
            }else // unsuccessful load
            {
                LOG(warning) << "Unable to load " << document << " with " << fg_info->GetObjectName();
            }
        }
        else // timeout        
        {
            LOG(warning) << "Timeout while loading " << document << " with " << fg_info->GetObjectName() << " after waiting " << fg_info->LoadTimeout() << " ms";
        }
    }
    return false;
    //auto max = std::max_element(frame_grabber_priorities.begin(), frame_grabber_priorities.end());
    //auto idx = std::distance(frame_grabber_priorities.begin(), max);
    //frame_grabber = shared_ptr<IFrameGrabber>(valid_frame_grabbers[idx]->Construct());
    //frame_grabber->InitializeFrameGrabber(this);

    //return frame_grabber->LoadFile(document);
}
bool DataStream::CanLoadDocument(const std::string& document)
{
    auto constructors = ObjectManager::Instance().GetConstructorsForInterface(IID_FrameGrabber);
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
                if (fg_info->CanLoadDocument(document))
                {
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(fg_info->Priority());
                }
            }
        }
    }
    return !valid_frame_grabbers.empty();
}

void DataStream::AddNode(shared_ptr<Nodes::Node> node)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    node->SetDataStream(this);
    top_level_nodes.push_back(node);
    dirty_flag = true;
}
void DataStream::AddNode(std::vector<shared_ptr<Nodes::Node>> nodes)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    for(auto& node: nodes)
    {
        node->SetDataStream(this);
        top_level_nodes.push_back(node);
    }
    dirty_flag = true;
}

void DataStream::RemoveNode(shared_ptr<Nodes::Node> node)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    auto itr = std::find(top_level_nodes.begin(), top_level_nodes.end(), node);
    if(itr != top_level_nodes.end())
    {
        top_level_nodes.erase(itr);
    }
}

void DataStream::LaunchProcess()
{
    StopProcess();
    processing_thread = boost::thread(boost::bind(&DataStream::process, this));
}

void DataStream::StopProcess()
{
    processing_thread.interrupt();
    processing_thread.join();
}


void DataStream::PauseProcess()
{
    paused = true;
}

void DataStream::ResumeProcess()
{
    paused = false;
}

void DataStream::process()
{
    cv::cuda::Stream streams[2];
    dirty_flag = true;
    int iteration_count = 0;
    signal_manager->register_thread(Signals::ANY);
    
    rmt_SetCurrentThreadName("DataStreamThread");
    auto node_update_connection = signal_manager->Connect<void(EagleLib::Nodes::Node*)>("NodeUpdated",
        std::bind([this](EagleLib::Nodes::Node* node)->void
        {
            dirty_flag = true;
        }, std::placeholders::_1), this, stream_id);

    auto update_connection = signal_manager->Connect<void()>("update",
        std::bind([this]()->void
    {
        dirty_flag = true;
    }), this, stream_id);

    while(!boost::this_thread::interruption_requested())
    {
        if(!paused)
        {
            Signals::thread_specific_queue::run();

            if (frame_grabber != nullptr)
            {
                if (dirty_flag)
                {
                    dirty_flag = false;
                    TS<SyncedMemory> current_frame;
                    std::vector<shared_ptr<Nodes::Node>> current_nodes;
                    {
                        std::lock_guard<std::mutex> lock(nodes_mtx);
                        current_frame = frame_grabber->GetNextFrame(streams[iteration_count % 2]);
                        current_nodes = top_level_nodes;
                    }
                    for (auto& node : current_nodes)
                    {
                        if(node->pre_check(current_frame))
                            node->process(current_frame, streams[iteration_count % 2]);
                    }
                    ++iteration_count;
                    
                }
                else
                {
                    // No update to parameters or variables since last run

                }
            }
            else
            {
                // Thread was launched without a frame grabber

            }
        }else
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
        }
    }
}

// **********************************************************************
//              DataStreamManager
// **********************************************************************
DataStreamManager* DataStreamManager::instance()
{
    static DataStreamManager* inst;
    if (inst == nullptr)
        inst = new DataStreamManager();
    return inst;
}

DataStreamManager::DataStreamManager()
{

}

DataStreamManager::~DataStreamManager()
{

}

std::shared_ptr<DataStream> DataStreamManager::create_stream()
{
    std::shared_ptr<DataStream> stream(new DataStream);
    stream->stream_id = streams.size();
    streams.push_back(stream);
    return stream;
}
void DataStreamManager::destroy_stream(DataStream* stream)
{
    for(auto itr = streams.begin(); itr != streams.end(); ++itr)
    {
        if(itr->get() == stream)
        {
            streams.erase(itr);
            return;
        }        
    }
}

DataStream* DataStreamManager::get_stream(size_t id)
{
    CV_Assert(id < streams.size());
    return streams[id].get();
}
