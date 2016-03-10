#include "DataStreamManager.h"
#include <opencv2/core.hpp>
#include "rcc/SystemTable.hpp"
#include "rcc/ObjectManager.h"
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include "utilities/sorting.hpp"
#include "EagleLib/Logging.h"
#include "Remotery.h"
#include "VariableManager.h"
using namespace EagleLib;

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
    LOG(error) << err.what();										    \
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
	_sig_manager = GetSignalManager();
    auto table = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table)
    {
        auto global_signal_manager = table->GetSingleton<SignalManager>();
		if (!global_signal_manager)
        {
			global_signal_manager  = SignalManager::get_instance();
			table->SetSingleton<SignalManager>(global_signal_manager);
			Signals::signal_manager::set_instance(global_signal_manager);
        }
		connections.push_back(global_signal_manager->connect<void(void)>("StopThreads", std::bind(&DataStream::PauseProcess, this), this));
		connections.push_back(global_signal_manager->connect<void(void)>("StartThreads", std::bind(&DataStream::ResumeProcess, this), this));
    }
	//connections.push_back(GetSignalManager()->connect<void(void)>("StopThreads", std::bind(&DataStream::StopProcess, this), this));
	//connections.push_back(GetSignalManager()->connect<void(void)>("StartThreads", std::bind(&DataStream::LaunchProcess, this), this));
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
	if (signal_manager == nullptr)
		signal_manager.reset(new SignalManager());
    return signal_manager.get();
}
std::shared_ptr<IVariableManager> DataStream::GetVariableManager()
{
    if(variable_manager == nullptr)
        variable_manager.reset(new VariableManager());
    return variable_manager;
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
        fg->Init(true);
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
				LOG(info) << "Loading " << document << " with frame_grabber: " << fg->GetTypeName() << " with priority: " << fg_info->Priority();
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
					LOG(trace) << fg_info->GetObjectName() << " can load document";
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(fg_info->Priority());
                }
            }
        }
    }
    return !valid_frame_grabbers.empty();
}

std::vector<shared_ptr<Nodes::Node>> DataStream::GetNodes()
{
    return top_level_nodes;
}

void DataStream::AddNode(shared_ptr<Nodes::Node> node)
{
	if (boost::this_thread::get_id() != processing_thread.get_id())
	{
		Signals::thread_specific_queue::push(std::bind(&DataStream::AddNode, this, node), processing_thread.get_id());
		return;
	}

    node->SetDataStream(this);
    node->Init(true);
    top_level_nodes.push_back(node);
    dirty_flag = true;
}
void DataStream::AddNodes(std::vector<shared_ptr<Nodes::Node>> nodes)
{
	if (boost::this_thread::get_id() != processing_thread.get_id())
	{
		Signals::thread_specific_queue::push(std::bind(&DataStream::AddNodes, this, nodes), processing_thread.get_id());
		return;
	}
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
	sig_StartThreads();
    processing_thread = boost::thread(boost::bind(&DataStream::process, this));
}

void DataStream::StopProcess()
{
    processing_thread.interrupt();
    processing_thread.join();
	sig_StopThreads();
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
    Signals::thread_registry::get_instance()->register_thread(Signals::ANY);
    rmt_SetCurrentThreadName("DataStreamThread");
    auto node_update_connection = signal_manager->connect<void(EagleLib::Nodes::Node*)>("NodeUpdated",
		std::bind([this](EagleLib::Nodes::Node* node)->void
		{
			dirty_flag = true;
		}, std::placeholders::_1), this);

    auto update_connection = signal_manager->connect<void()>("update",
        std::bind([this]()->void
        {
            dirty_flag = true;
        }), this);

    auto object_update_connection = signal_manager->connect<void(ParameteredObject*)>("parameter_updated",
        std::bind([this](ParameteredObject*)->void
        {
            dirty_flag = true;
        }, std::placeholders::_1), this);

	auto parameter_added_connection = signal_manager->connect<void(ParameteredObject*)>("parameter_added",
		std::bind([this](ParameteredObject*)->void
		{
			dirty_flag = true;
		}, std::placeholders::_1), this);

    LOG(info) << "Starting stream thread";
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
                        try
                        {
                            std::lock_guard<std::mutex> lock(nodes_mtx);
                            rmt_ScopedCPUSample(GrabbingFrame);
                            current_frame = frame_grabber->GetNextFrame(streams[iteration_count % 2]);
                            current_nodes = top_level_nodes;
                        }CATCH_MACRO
                    }
                    for (auto& node : current_nodes)
                    {
                        if(node->pre_check(current_frame))
                            node->process(current_frame, streams[iteration_count % 2]);
                    }
                    ++iteration_count;
                    if(!dirty_flag)
                        LOG(trace) << "Dirty flag not set and end of iteration " << iteration_count << " with frame number " << current_frame.frame_number;
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
    LOG(info) << "Stream thread shutting down";
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
