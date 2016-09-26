#include "EagleLib/nodes/IFrameGrabber.hpp"
#include "EagleLib/Logging.h"
#include "EagleLib/DataStreamManager.h"
#include "EagleLib/utilities/sorting.hpp"
#include "Remotery.h"
#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
#include <MetaObject/MetaObjectFactory.hpp>

#include "ISimpleSerializer.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;

int FrameGrabberInfo::LoadTimeout() const
{
    return 1000;
}

std::vector<std::string> FrameGrabberInfo::ListLoadableDocuments() const
{
    return std::vector<std::string>();
}
::std::vector<::std::string> IFrameGrabber::ListAllLoadableDocuments()
{
    std::vector<std::string> output;
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(IID_FrameGrabber);
    for(auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(auto fg_info = dynamic_cast<FrameGrabberInfo*>(info))
        {
            auto devices = fg_info->ListLoadableDocuments();
            output.insert(output.end(), devices.begin(), devices.end());
        }
    }
    return output;
}

rcc::shared_ptr<IFrameGrabber> IFrameGrabber::Create(const std::string& uri, const std::string& preferred_loader)
{
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(IID_FrameGrabber);
    std::vector<IObjectConstructor*> valid_constructors;
    std::vector<int> valid_constructor_priority;
    for(auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(auto fg_info = dynamic_cast<FrameGrabberInfo*>(info))
        {
            int priority = fg_info->CanLoadDocument(uri);
            if(priority != 0)
            {
                valid_constructors.push_back(constructor);
                valid_constructor_priority.push_back(priority);
            }
        }
    }
    if (valid_constructors.empty())
    {
        LOG(warning) << "No valid frame grabbers for " << uri;
        return rcc::shared_ptr<IFrameGrabber>();
    }

    auto idx = sort_index_descending(valid_constructor_priority);
    if (preferred_loader.size())
    {
        for (int i = 0; i < valid_constructors.size(); ++i)
        {
            if (preferred_loader == valid_constructors[i]->GetName())
            {
                idx.insert(idx.begin(), i);
                break;
            }
        }
    }

    for (int i = 0; i < idx.size(); ++i)
    {
        auto fg = rcc::shared_ptr<IFrameGrabber>(valid_constructors[idx[i]]->Construct());
        auto fg_info = dynamic_cast<FrameGrabberInfo*>(valid_constructors[idx[i]]->GetObjectInfo());
        //fg->InitializeFrameGrabber(this);
        fg->Init(true);
        
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
        obj->document = uri;
        auto future = obj->promise.get_future();
        static std::vector<boost::thread*> connection_threads;
        // TODO cleanup the connection threads


        boost::thread* connection_thread = new boost::thread([obj]()->void {
            try
            {
                obj->load();
            }
            catch (cv::Exception&e)
            {
                LOG(debug) << e.what();
            }

            delete obj;
        });
        
        if (connection_thread->timed_join(boost::posix_time::milliseconds(fg_info->LoadTimeout())))
        {
            if (future.get())
            {
                //top_level_nodes.emplace_back(fg);
                LOG(info) << "Loading " << uri << " with frame_grabber: " << fg->GetTypeName() << " with priority: " << valid_constructor_priority[idx[i]];
                delete connection_thread;
                return fg; // successful load
            }
            else // unsuccessful load
            {
                LOG(warning) << "Unable to load " << uri << " with " << fg_info->GetObjectName();
            }
        }
        else // timeout        
        {
            LOG(warning) << "Timeout while loading " << uri << " with " << fg_info->GetObjectName() << " after waiting " << fg_info->LoadTimeout() << " ms";
            connection_threads.push_back(connection_thread);
        }
    }
    return rcc::shared_ptr<IFrameGrabber>();
}



IFrameGrabber::IFrameGrabber()
{
    parent_stream = nullptr;
    this->_ctx = &ctx;
    ctx.stream = &stream;
}

std::string IFrameGrabber::GetSourceFilename()
{
    return loaded_document;
}

void IFrameGrabber::InitializeFrameGrabber(IDataStream* stream)
{
    parent_stream = stream;
    
    if(stream)
    {
        SetupSignals(stream->GetRelayManager());
        //update_signal = stream->GetSignalManager()->get_signal<void()>("update", this);
        SetupVariableManager(stream->GetVariableManager().get());
    }
}
void IFrameGrabber::Init(bool firstInit)
{
    IMetaObject::Init(firstInit);
    if(!firstInit)
    {
        //LoadFile(loaded_document); // each implementation should know if it needs to reload the file on recompile
    }
}
bool IFrameGrabber::ProcessImpl()
{
    auto frame = GetNextFrame(*_ctx->stream);
    if (!frame.empty())
    {
        this->current_frame_param.UpdateData(frame, frame.frame_number, _ctx);
        return true;
    }
    return false;
}
void IFrameGrabber::Serialize(ISimpleSerializer* pSerializer)
{
    IMetaObject::Serialize(pSerializer);
    SERIALIZE(loaded_document);
    SERIALIZE(parent_stream);
}
FrameGrabberBuffered::FrameGrabberBuffered():
    IFrameGrabber()
{
    buffer_begin_frame_number = 0;
    buffer_end_frame_number = 0;
    playback_frame_number = 0;
    _is_stream = false;
}
FrameGrabberBuffered::~FrameGrabberBuffered()
{
    
}

FrameGrabberThreaded::~FrameGrabberThreaded()
{
    buffer_thread.interrupt();
    buffer_thread.join();
}

/*TS<SyncedMemory> FrameGrabberBuffered::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return GetFrame(playback_frame_number, stream);
}*/

TS<SyncedMemory> FrameGrabberBuffered::GetFrame(int index, cv::cuda::Stream& stream)
{
    while(frame_buffer.empty())
    {
        // Wait for frame
        //boost::this_thread::interruptible_wait(10);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }
    for(auto& itr: frame_buffer)
    {
        if(itr.frame_number == index)
        {
            return itr;
        }
    }
    return TS<SyncedMemory>();
}

TS<SyncedMemory> FrameGrabberBuffered::GetNextFrame(cv::cuda::Stream& stream)
{

    boost::mutex::scoped_lock bLock(buffer_mtx);
    // Waiting on the grabbing thread to load frames
    int attempts = 0;
    while(playback_frame_number > buffer_end_frame_number - 5 )
    {
        LOG(trace) << "Playback frame number (" << playback_frame_number << ") is too close to end of frame buffer (" << buffer_end_frame_number << ") - waiting for new frame to be read";
        frame_grabbed_cv.wait_for(bLock, boost::chrono::milliseconds(10));
        if(attempts > 500)
            return TS<SyncedMemory>();
    }
    int index = 0;
    long long desired_frame;
    if (_is_stream)
    {
        desired_frame = std::max(int(buffer_end_frame_number - 5), int(buffer_begin_frame_number));
        if (desired_frame == playback_frame_number)
            return TS<SyncedMemory>();
    }
    else
    {
        desired_frame = playback_frame_number + 1;
    }
 
   for(auto& itr : frame_buffer)
    {
        if(itr.frame_number == desired_frame)
        {
            LOG(trace) << "Found next frame in frame buffer with frame index (" << desired_frame << ") at buffer index (" << index << ")";
            playback_frame_number = desired_frame;
            // Found the next frame
            //if (update_signal)
                //(*update_signal)();
            sig_update();
            return itr;
        }
        ++index;
    }
    // If we get to this point, perhaps a frame was dropped. look for the next valid frame number in the frame buffer
   if(_is_stream)
   {
        if(desired_frame < buffer_begin_frame_number)
        {
            // If this is a live stream and we've fallen behind, rail desired frame and return back of frame buffer
            playback_frame_number = frame_buffer.begin()->frame_number;
            sig_update();
            return *frame_buffer.begin();        
        }   
   }
    LOG(trace) << "Unable to find desired frame (" << desired_frame << ") in frame buffer [" << buffer_begin_frame_number << "," << buffer_end_frame_number << "]";
    for(int i = 0; i < frame_buffer.size(); ++i)
    {
        if(frame_buffer[i].frame_number == playback_frame_number)
        {
            if(i < frame_buffer.size() - 1)
            {
                LOG(trace) << "Frame (" << playback_frame_number << ") was dropped, next valid frame number: " << frame_buffer[i+1].frame_number;
                playback_frame_number = frame_buffer[i + 1].frame_number;
                return frame_buffer[i+1];
            }
        }
    }
    // If we get to this point then perhaps playback frame number 
    LOG(trace) << "Unable to find valid frame in frame buffer";
    return TS<SyncedMemory>();
}
TS<SyncedMemory> FrameGrabberBuffered::GetFrameRelative(int index, cv::cuda::Stream& stream)
{
    boost::mutex::scoped_lock bLock(buffer_mtx);
    int _index = 0;
    for (auto& itr : frame_buffer)
    {
        if (itr.frame_number == playback_frame_number + index)
        {
            LOG(trace) << "Found next frame in frame buffer with frame index (" << playback_frame_number + index << ") at buffer index (" << _index << ")";
            return itr;
        }
        ++_index;
    }
    LOG(trace) << "Unable to find requested frame (" << playback_frame_number + index << ") in frame buffer. [" << frame_buffer.front().frame_number << "," << frame_buffer.back().frame_number << "]";
    return TS<SyncedMemory>();
}

long long FrameGrabberBuffered::GetFrameNumber()
{
    return playback_frame_number;
}
void FrameGrabberBuffered::PushFrame(TS<SyncedMemory> frame, bool blocking)
{
    boost::mutex::scoped_lock bLock(buffer_mtx);

    // Waiting for the reading thread to catch up
    if(blocking)
    {
        while(((buffer_begin_frame_number + 5 > playback_frame_number && frame_buffer.size() == frame_buffer.capacity()) && !_is_stream))
        {
            LOG(trace) << "Frame buffer is full and playback frame (" << playback_frame_number << ") is too close to the beginning of the frame buffer (" << buffer_begin_frame_number << ")";
            frame_read_cv.wait_for(bLock, boost::chrono::milliseconds(10));
        }
    }
    
    buffer_end_frame_number = frame.frame_number;
    frame_buffer.push_back(frame);
    if(frame_buffer.size())
        buffer_begin_frame_number = frame_buffer[0].frame_number;
    frame_grabbed_cv.notify_all();
    sig_update();
}
void FrameGrabberThreaded::Buffer()
{
    cv::cuda::Stream read_stream;
    rmt_SetCurrentThreadName("FrameGrabberThread");
    LOG(info) << "Starting buffer thread";
    while(!boost::this_thread::interruption_requested())
    {   
        while(_pause)
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        try
        {
            TS<SyncedMemory> frame;
            {
                rmt_ScopedCPUSample(BufferingFrame);
                boost::mutex::scoped_lock gLock(grabber_mtx);
                frame = GetNextFrameImpl(read_stream);
            }            
            if (!frame.empty())
            {
                PushFrame(frame, true);
            }else
            {
                LOG(trace) << "Read empty frame from frame grabber";
            }
        }catch(cv::Exception& e)
        {
            LOG(warning) << "Error reading next frame: " << e.what();
        }
    }
    LOG(info) << "Shutting down buffer thread";
}

void FrameGrabberThreaded::StartThreads()
{
    if(_pause)
    {
        _pause = false;
        return;
    }
    LOG(info);
    StopThreads();
    buffer_thread = boost::thread(boost::bind(&FrameGrabberThreaded::Buffer, this));
}

void FrameGrabberThreaded::StopThreads()
{
    LOG(info);
    buffer_thread.interrupt();
    DOIF_LOG_FAIL(buffer_thread.joinable(), buffer_thread.join(), trace);
}

void FrameGrabberThreaded::PauseThreads()
{
    _pause = true;
}

void FrameGrabberThreaded::ResumeThreads()
{
    _pause = false;
}


std::string FrameGrabberInfo::Print() const
{
    std::stringstream ss;
    ss << NodeInfo::Print();
    auto documents = ListLoadableDocuments();
    if(documents.size())
    {
        ss << "\n------- Loadable Documents ---------\n";
        for(auto& doc : documents)
        {
            ss << "  " << doc << "\n";
        }
    }
    ss << "Load timeout: " << LoadTimeout() << "\n";
    return ss.str();
}


void FrameGrabberBuffered::Init(bool firstInit)
{
    IFrameGrabber::Init(firstInit);
    
    frame_buffer.set_capacity(frame_buffer_size);
    
    /*boost::function<void(cv::cuda::Stream*)> f =[&](cv::cuda::Stream*)
    {
        boost::mutex::scoped_lock lock(buffer_mtx);
        frame_buffer.set_capacity(frame_buffer_size);
    };*/
    //RegisterParameterCallback("Frame buffer size", f, true, true);
}

void FrameGrabberThreaded::Init(bool firstInit)
{
    FrameGrabberBuffered::Init(firstInit);
    StartThreads();
}
void FrameGrabberBuffered::Serialize(ISimpleSerializer* pSerializer)
{
    IFrameGrabber::Serialize(pSerializer);
    SERIALIZE(frame_buffer);
}
