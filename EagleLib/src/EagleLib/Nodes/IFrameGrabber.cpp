#include "EagleLib/Nodes/IFrameGrabber.hpp"
#include "EagleLib/IDataStream.hpp"
#include "EagleLib/utilities/sorting.hpp"
#include "EagleLib/Nodes/FrameGrabberInfo.hpp"
#include <EagleLib/ICoordinateManager.h>
#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Logging/Profiling.hpp>
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
#include <MetaObject/MetaObjectFactory.hpp>
#include <MetaObject/Logging/Profiling.hpp>
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
    auto constructors = mo::MetaObjectFactory::Instance()->
            GetConstructors(EagleLib::Nodes::IFrameGrabber::s_interfaceID);
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

rcc::shared_ptr<IFrameGrabber> IFrameGrabber::Create(const std::string& uri,
                                                     const std::string& preferred_loader)
{
    auto constructors = mo::MetaObjectFactory::Instance()->
            GetConstructors(EagleLib::Nodes::IFrameGrabber::s_interfaceID);
    std::vector<IObjectConstructor*> valid_constructors;
    std::vector<int> valid_constructor_priority;
    for(auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(auto fg_info = dynamic_cast<FrameGrabberInfo*>(info))
        {
            int priority = fg_info->CanLoadDocument(uri);
            LOG(debug) << fg_info->GetDisplayName() << " priority: " << priority;
            if(priority != 0)
            {
                valid_constructors.push_back(constructor);
                valid_constructor_priority.push_back(priority);
            }
        }
    }
    if (valid_constructors.empty())
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

        LOG(warning) << "No valid frame grabbers for " << uri
                     << " framegrabbers: " << f();
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
                LOG(info) << "Loading " << uri << " with frame_grabber: " << fg->GetTypeName() << " with priority: " << valid_constructor_priority[idx[i]];
                delete connection_thread;
                fg->loaded_document = uri;
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
    //this->_ctx = &ctx;
    //ctx.stream = &stream;
}
void IFrameGrabber::on_loaded_document_modified(mo::Context*, mo::IParameter*)
{
    this->LoadFile(loaded_document);
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
void IFrameGrabber::Restart()
{
    this->LoadFile(this->GetSourceFilename());
}
void IFrameGrabber::Init(bool firstInit)
{
    Node::Init(firstInit);
    if(!firstInit)
    {
        //LoadFile(loaded_document); // each implementation should know if it needs to reload the file on recompile
    }
}
bool IFrameGrabber::ProcessImpl()
{
    auto frame = GetNextFrame(Stream());
    if (!frame.empty())
    {
        this->_modified = true;
        this->current_frame_param.UpdateData(frame, frame.frame_number, _ctx);
        return true;
    }
    return false;
}
void IFrameGrabber::Serialize(ISimpleSerializer* pSerializer)
{
    Node::Serialize(pSerializer);
    SERIALIZE(loaded_document);
    SERIALIZE(parent_stream);
}
FrameGrabberBuffered::FrameGrabberBuffered():
    IFrameGrabber()
{
    buffer_begin_frame_number = 0;
    buffer_end_frame_number = 0;
    playback_frame_number = -1;
    _is_stream = false;
}
FrameGrabberBuffered::~FrameGrabberBuffered()
{
    
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
        LOG_EVERY_N(debug, 100) << "Playback frame number (" << playback_frame_number
                                << ") is too close to end of frame buffer ("
                                << buffer_end_frame_number << ") - waiting for new frame to be read";
        frame_grabbed_cv.wait_for(bLock, boost::chrono::nanoseconds(10));
        if(attempts > 500)
            return TS<SyncedMemory>();
        ++attempts;
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
        if(_frame_number_playback_queue.size())
        {
            playback_frame_number = _frame_number_playback_queue.front();
            _frame_number_playback_queue.pop();
            LOG(trace) << "Got next frame index from playback queue " << playback_frame_number;
        }
        desired_frame = playback_frame_number;
    }

    for(auto& itr : frame_buffer)
    {
        if(itr.frame_number == desired_frame)
        {
            LOG(trace) << "Found next frame in frame buffer with frame index ("
                       << desired_frame << ") at buffer index (" << index << ")";
            playback_frame_number = desired_frame;
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
TS<SyncedMemory> FrameGrabberBuffered::GetCurrentFrame(cv::cuda::Stream& stream)
{
    boost::mutex::scoped_lock bLock(buffer_mtx);
    if(frame_buffer.size())
    {
        return frame_buffer.back();
    }
    return TS<SyncedMemory>();
}
void FrameGrabberBuffered::PushFrame(TS<SyncedMemory> frame, bool blocking)
{
    //SCOPED_PROFILE_NODE
    boost::mutex::scoped_lock bLock(buffer_mtx);

    // Waiting for the reading thread to catch up
    if(blocking)
    {
        while((buffer_begin_frame_number + 5 > playback_frame_number && 
            frame_buffer.size() == frame_buffer.capacity()) && 
            !_is_stream)
        {
            LOG(trace) << "Frame buffer is full and playback frame (" << playback_frame_number << ") is too close to the beginning of the frame buffer (" << buffer_begin_frame_number << ")";
            sig_update(); // Due to threading, sometimes the threads can get out of sync and the reading thread thinks there isn't new data to read
            _modified = true;
            frame_read_cv.wait_for(bLock, boost::chrono::milliseconds(10));
        }
    }
    
    buffer_end_frame_number = frame.frame_number;
    frame_buffer.push_back(frame);
    _frame_number_playback_queue.push(frame.frame_number);
    if(frame_buffer.size())
        buffer_begin_frame_number = frame_buffer[0].frame_number;
    frame_grabbed_cv.notify_all();
    sig_update();
    _modified = true;
}
int FrameGrabberThreaded::Buffer()
{
    try
    {
        TS<SyncedMemory> frame;
        {
            boost::mutex::scoped_lock gLock(grabber_mtx);
            frame = GetNextFrameImpl(_buffer_thread_handle.GetContext()->GetStream());
        }
        if (!frame.empty())
        {
            PushFrame(frame, true);
            _empty_frame_count = 0;
            return 0;
        }
        else
        {
            ++_empty_frame_count;
            LOG_EVERY_N(warning, 500) << "Read empty frame from frame grabber";
            if(_empty_frame_count > 100)
            {
                // Haven't received a new frame in over 3 seconds.
                // Signal end of stream
                sig_eos();
            }
        }
    }
    catch (cv::Exception& e)
    {
        LOG(warning) << "Error reading next frame: " << e.what();
    }
    return 30;
}

void FrameGrabberThreaded::StartThreads()
{
    //StopThreads();
    LOG(info) << "Starting buffer thread";
    _buffer_thread_handle.Start();
}

void FrameGrabberThreaded::StopThreads()
{
    LOG(info) << "Stopping buffer thread";
    _buffer_thread_handle.Stop();
}

FrameGrabberThreaded::FrameGrabberThreaded()
{
    _empty_frame_count = 0;
}

void FrameGrabberThreaded::Init(bool firstinit)
{
    FrameGrabberBuffered::Init(firstinit);
    _buffer_thread_handle.SetInnerLoop(this->GetSlot_Buffer<int(void)>());
    _buffer_thread_handle.SetThreadName("FrameGrabberBufferThread");
}

void FrameGrabberThreaded::PauseThreads()
{
    _buffer_thread_handle.Stop();
}

void FrameGrabberThreaded::ResumeThreads()
{
    _buffer_thread_handle.Start();
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

void FrameGrabberBuffered::Serialize(ISimpleSerializer* pSerializer)
{
    IFrameGrabber::Serialize(pSerializer);
    SERIALIZE(frame_buffer);
}

class TestFrameGrabber: public IFrameGrabber
{
public:
    MO_DERIVE(TestFrameGrabber, IFrameGrabber)
    MO_END;
    TestFrameGrabber()
    {
        cv::Mat output_(640,480, CV_8UC3);
        cv::randn(output_, 128, 10);
        this->output = TS<SyncedMemory>(0.0, (long long)0, output_);
    }
    bool LoadFile(const ::std::string& file_path)
    {
        return true;
    }
    long long GetFrameNumber()
    {
        return 0;
    }
    long long GetNumFrames()
    {
        return 0;
    }
    TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream)
    {
        return output;
    }
    TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream)
    {
        return output;
    }
    TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream)
    {
        return output;
    }
    
    TS<SyncedMemory> GetFrameRelative(int index, cv::cuda::Stream& stream)
    {
        return output;
    }

    rcc::shared_ptr<ICoordinateManager> GetCoordinateManager()
    {
        return rcc::shared_ptr<ICoordinateManager>();
    }
    TS<SyncedMemory> output;
    static int CanLoadDocument(const std::string& doc)
    {
        return -1;
    }
};

MO_REGISTER_CLASS(TestFrameGrabber)
