#include "DataStreamManager.h"
#include "rcc/SystemTable.hpp"
#include "rcc/ObjectManager.h"
#include "utilities/sorting.hpp"
#include "EagleLib/Logging.h"
#include "Remotery.h"

#include "ParameterBuffer.h"
#include "IVariableSink.h"
#include "IViewManager.h"
#include "ICoordinateManager.h"
#include "rendering/RenderingEngine.h"
#include "tracking/ITrackManager.h"
#include "frame_grabber_base.h"
#include "nodes/Node.h"
#include "nodes/NodeManager.h"

#include <opencv2/core.hpp>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>

#include <signals/boost_thread.h>
#include <parameters/VariableManager.h>


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


IDataStream::~IDataStream()
{

}
namespace EagleLib
{
	class DataStream: public IDataStream
	{
	public:
		DataStream();
		virtual ~DataStream();

		// Handles user interactions such as moving the viewport, user interface callbacks, etc
		virtual rcc::weak_ptr<IViewManager>            GetViewManager();

		// Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
		virtual rcc::weak_ptr<ICoordinateManager>      GetCoordinateManager();

		// Handles actual rendering of data.  Use for adding extra objects to the scene
		virtual rcc::weak_ptr<IRenderEngine>           GetRenderingEngine();

		// Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
		virtual rcc::weak_ptr<ITrackManager>            GetTrackManager();

		// Handles actual loading of the image, etc
		virtual rcc::weak_ptr<IFrameGrabber>           GetFrameGrabber();

		virtual std::shared_ptr<Parameters::IVariableManager> GetVariableManager();

		virtual SignalManager*							GetSignalManager();

		virtual IParameterBuffer*						GetParameterBuffer();

		virtual std::vector<rcc::shared_ptr<Nodes::Node>> GetNodes();

		virtual bool LoadDocument(const std::string& document, const std::string& prefered_loader = "");
	
		virtual std::vector<rcc::shared_ptr<Nodes::Node>> AddNode(const std::string& nodeName);
		virtual void AddNode(rcc::shared_ptr<Nodes::Node> node);
		virtual void AddNodes(std::vector<rcc::shared_ptr<Nodes::Node>> node);
		virtual void RemoveNode(rcc::shared_ptr<Nodes::Node> node);
		virtual void RemoveNode(Nodes::Node* node);
		//void StartThread();
		//void StopThread();
		//void PauseThread();
		//void ResumeThread();
		void process();

		void AddVariableSink(IVariableSink* sink);
		void RemoveVariableSink(IVariableSink* sink);

	protected:
		int stream_id;
		size_t _thread_id;
		rcc::shared_ptr<IViewManager>							view_manager;
		rcc::shared_ptr<ICoordinateManager>						coordinate_manager;
		rcc::shared_ptr<IRenderEngine>							rendering_engine;
		rcc::shared_ptr<ITrackManager>							track_manager;
		rcc::shared_ptr<IFrameGrabber>							frame_grabber;
		//std::shared_ptr<Parameters::IVariableManager>			variable_manager;
		std::shared_ptr<SignalManager>							signal_manager;
		std::vector<rcc::shared_ptr<Nodes::Node>>				top_level_nodes;
		std::shared_ptr<IParameterBuffer>						_parameter_buffer;
		std::mutex    											nodes_mtx;
		bool													paused;
		cv::cuda::Stream										cuda_stream;
		boost::thread											processing_thread;
		volatile bool											dirty_flag;
		std::vector<std::shared_ptr<Signals::connection>>		connections;
		cv::cuda::Stream										streams[2];
		std::vector<IVariableSink*>                             variable_sinks;
	public:
		SIGNALS_BEGIN(DataStream, IDataStream);
			SIG_SEND(StartThreads);
			SIG_SEND(StopThreads);
			SLOT_DEF(void, StartThread);
			REGISTER_SLOT(StartThread);
			SLOT_DEF(void, StopThread);
			REGISTER_SLOT(StopThread);
			SLOT_DEF(void, PauseThread);
			REGISTER_SLOT(PauseThread);
			SLOT_DEF(void, ResumeThread);
			REGISTER_SLOT(ResumeThread);
		SIGNALS_END
	};
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
		connections.push_back(global_signal_manager->connect<void(void)>("StopThreads", std::bind(&DataStream::StopThread, this), this));
		connections.push_back(global_signal_manager->connect<void(void)>("StartThreads", std::bind(&DataStream::StartThread, this), this));
		connections.push_back(global_signal_manager->connect<void(void)>("PauseThreads", std::bind(&DataStream::PauseThread, this), this));
		connections.push_back(global_signal_manager->connect<void(void)>("ResumeThreads", std::bind(&DataStream::ResumeThread, this), this));
    }
	this->setup_signals(GetSignalManager());
    paused = false;
    stream_id = 0;
	_thread_id = 0;
}

DataStream::~DataStream()
{
	StopThread();
	top_level_nodes.clear();
	frame_grabber.reset();
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

// Handles actual loading of the image, etc
rcc::weak_ptr<IFrameGrabber> DataStream::GetFrameGrabber()
{
    return frame_grabber;
}

SignalManager* DataStream::GetSignalManager()
{
	if (signal_manager == nullptr)
		signal_manager.reset(new SignalManager());
    return signal_manager.get();
}
IParameterBuffer* DataStream::GetParameterBuffer()
{
	if (_parameter_buffer == nullptr)
		_parameter_buffer.reset(new ParameterBuffer(10));
	return _parameter_buffer.get();

}
std::shared_ptr<Parameters::IVariableManager> DataStream::GetVariableManager()
{
    if(_variable_manager == nullptr)
        _variable_manager.reset(new Parameters::VariableManager());
	return _variable_manager;
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
        LOG(warning) << "No valid frame grabbers for " << file_to_load;
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
        auto fg_info = static_cast<FrameGrabberInfo*>(valid_frame_grabbers[idx[i]]->GetObjectInfo());
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
        boost::thread connection_thread = boost::thread([obj]()->void{
            try
            {
                obj->load();
            }catch(cv::Exception&e)
            {
                LOG(debug) << e.what();
            }
            
            delete obj;
        });
        if(connection_thread.timed_join(boost::posix_time::milliseconds(fg_info->LoadTimeout())))
        {
            if(future.get())
            {
                frame_grabber = fg;
                LOG(info) << "Loading " << file_to_load << " with frame_grabber: " << fg->GetTypeName() << " with priority: " << frame_grabber_priorities[idx[i]];
                return true; // successful load
            }else // unsuccessful load
            {
                LOG(warning) << "Unable to load " << file_to_load << " with " << fg_info->GetObjectName();
            }
        }
        else // timeout        
        {
            LOG(warning) << "Timeout while loading " << file_to_load << " with " << fg_info->GetObjectName() << " after waiting " << fg_info->LoadTimeout() << " ms";
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
                int priority = fg_info->CanLoadDocument(doc_to_load);
                if (priority != 0)
                {
					LOG(trace) << fg_info->GetObjectName() << " can load document";
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(priority);
                }
            }
        }
    }
    return !valid_frame_grabbers.empty();
}

std::vector<rcc::shared_ptr<Nodes::Node>> DataStream::GetNodes()
{
    return top_level_nodes;
}
std::vector<rcc::shared_ptr<Nodes::Node>> DataStream::AddNode(const std::string& nodeName)
{
	return EagleLib::NodeManager::getInstance().addNode(nodeName, this);
}
void DataStream::AddNode(rcc::shared_ptr<Nodes::Node> node)
{
	node->SetDataStream(this);
	node->Init(true);
	if (boost::this_thread::get_id() != processing_thread.get_id() && !paused  && _thread_id != 0)
	{
        Signals::thread_specific_queue::push(std::bind(static_cast<void(DataStream::*)(rcc::shared_ptr<Nodes::Node>)>(&DataStream::AddNode), this, node), _thread_id);
		return;
	}
    top_level_nodes.push_back(node);
    dirty_flag = true;
}
void DataStream::AddNodes(std::vector<rcc::shared_ptr<Nodes::Node>> nodes)
{
	for (auto& node : nodes)
	{
		node->SetDataStream(this);
		node->Init(true);
	}
	if (boost::this_thread::get_id() != processing_thread.get_id() && _thread_id != 0 && !paused)
	{
        Signals::thread_specific_queue::push(std::bind(&DataStream::AddNodes, this, nodes), _thread_id);
		return;
	}
    for(auto& node: nodes)
    {
        top_level_nodes.push_back(node);
    }
    dirty_flag = true;
}
void DataStream::RemoveNode(Nodes::Node* node)
{
	std::lock_guard<std::mutex> lock(nodes_mtx);
	auto itr = std::find(top_level_nodes.begin(), top_level_nodes.end(), node);
	if(itr != top_level_nodes.end())
	{
		top_level_nodes.erase(itr);
	}
}
void DataStream::RemoveNode(rcc::shared_ptr<Nodes::Node> node)
{
    std::lock_guard<std::mutex> lock(nodes_mtx);
    auto itr = std::find(top_level_nodes.begin(), top_level_nodes.end(), node);
    if(itr != top_level_nodes.end())
    {
        top_level_nodes.erase(itr);
    }
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
    processing_thread = boost::thread(boost::bind(&DataStream::process, this));
}

void DataStream::StopThread()
{
    processing_thread.interrupt();
    processing_thread.join();
	sig_StopThreads();
	LOG(trace);
}


void DataStream::PauseThread()
{
    paused = true;
	sig_StopThreads();
	LOG(trace);
}

void DataStream::ResumeThread()
{
    paused = false;
	sig_StartThreads();
	LOG(trace);
}

void DataStream::process()
{
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

    auto object_update_connection = signal_manager->connect<void(Parameters::ParameteredObject*)>("parameter_updated",
        std::bind([this](Parameters::ParameteredObject*)->void
        {
            dirty_flag = true;
        }, std::placeholders::_1), this);

	auto parameter_added_connection = signal_manager->connect<void(Parameters::ParameteredObject*)>("parameter_added",
		std::bind([this](Parameters::ParameteredObject*)->void
		{
			dirty_flag = true;
		}, std::placeholders::_1), this);

	if(_thread_id == 0)
		_thread_id = Signals::get_this_thread();

    LOG(info) << "Starting stream thread";
    while(!boost::this_thread::interruption_requested())
    {
        if(!paused)
        {
            Signals::thread_specific_queue::run(_thread_id);

			if(dirty_flag)
			{
				dirty_flag = false;
				TS<SyncedMemory> current_frame;
				std::vector<rcc::shared_ptr<Nodes::Node>> current_nodes;
				{
					std::lock_guard<std::mutex> lock(nodes_mtx);
					current_nodes = top_level_nodes;
				}
				if (frame_grabber != nullptr)
				{
					try
					{
						rmt_ScopedCPUSample(GrabbingFrame);
						current_frame = frame_grabber->GetNextFrame(streams[iteration_count % 2]);
								
					}CATCH_MACRO		
				}
				for (auto& node : current_nodes)
				{
					if(node->pre_check(current_frame))
					{
						try
						{
							node->process(current_frame, streams[iteration_count % 2]);
						}CATCH_MACRO
					}
				}
				for(auto sink : variable_sinks)
				{
					sink->SerializeVariables(current_frame.frame_number, _variable_manager.get());
				}
				++iteration_count;
				if(!dirty_flag)
					LOG(trace) << "Dirty flag not set and end of iteration " << iteration_count << " with frame number " << current_frame.frame_number;
			}
        }else
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
        }
    }
    LOG(info) << "Stream thread shutting down";
}

IDataStream::Ptr IDataStream::create(const std::string& document, const std::string& preferred_frame_grabber)
{
	auto stream = ObjectManager::Instance().GetObject<IDataStream, IID_DataStream>("DataStream");
    if(document.size() == 0)
        return stream;
	if(stream->LoadDocument(document, preferred_frame_grabber))
	{
		return stream;
	}
	return IDataStream::Ptr();
}

REGISTERCLASS(DataStream)