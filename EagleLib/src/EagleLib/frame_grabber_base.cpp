#include "frame_grabber_base.h"
#include "EagleLib/Logging.h"
#include "EagleLib/DataStreamManager.h"
#include "Remotery.h"
#include "EagleLib/ParameteredObjectImpl.hpp"
#include <signals/logging.hpp>
using namespace EagleLib;
IFrameGrabber::IFrameGrabber()
{
    update_signal = nullptr;
    parent_stream = nullptr;
}

std::string IFrameGrabber::GetSourceFilename()
{
    return loaded_document;
}
void IFrameGrabber::InitializeFrameGrabber(DataStream* stream)
{
    parent_stream = stream;
	
    if(stream)
    {
		setup_signals(stream->GetSignalManager());
		update_signal = stream->GetSignalManager()->get_signal<void()>("update", this);
        SetupVariableManager(stream->GetVariableManager().get());
    }
}
void IFrameGrabber::Serialize(ISimpleSerializer* pSerializer)
{
    ParameteredIObject::Serialize(pSerializer);
    SERIALIZE(loaded_document);
    SERIALIZE(parent_stream);
}
void IFrameGrabber::Init(bool firstInit)
{
    if(!firstInit)
    {
        update_signal = parent_stream->GetSignalManager()->get_signal<void()>("update", this);
    }
}
FrameGrabberBuffered::FrameGrabberBuffered()
{
    buffer_begin_frame_number = 0;
    buffer_end_frame_number = 0;
    playback_frame_number = 0;
    
}
void FrameGrabberBuffered::InitializeFrameGrabber(DataStream* stream)
{
    IFrameGrabber::InitializeFrameGrabber(stream);
    if (stream)
    {
        _callback_connections.push_back(stream->GetSignalManager()->connect<void()>("StartThreads", std::bind(&FrameGrabberBuffered::LaunchBufferThread, this), this));
        _callback_connections.push_back(stream->GetSignalManager()->connect<void()>("StopThreads", std::bind(&FrameGrabberBuffered::StopBufferThread, this), this));
    }
    LaunchBufferThread();
}
FrameGrabberBuffered::~FrameGrabberBuffered()
{
    buffer_thread.interrupt();
    buffer_thread.join();
}
TS<SyncedMemory> FrameGrabberBuffered::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return GetFrame(playback_frame_number, stream);
}

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
    while(playback_frame_number > buffer_end_frame_number - 5 )
    {
        LOG(trace) << "Playback frame number (" << playback_frame_number << ") is too close to end of frame buffer (" << buffer_end_frame_number << ") - waiting for new frame to be read";
        frame_grabbed_cv.wait_for(bLock, boost::chrono::milliseconds(10));
    }
    int index = 0;
    for(auto& itr : frame_buffer)
    {
        if(itr.frame_number == playback_frame_number + 1)
        {
            LOG(trace) << "Found next frame in frame buffer with frame index (" << playback_frame_number + 1 << ") at buffer index (" << index << ")";
            playback_frame_number++;
            // Found the next frame
            if (update_signal)
                (*update_signal)();
            return itr;
        }
        ++index;
    }
    // If we get to this point, perhaps a frame was dropped. look for the next valid frame number in the frame buffer
    LOG(trace) << "Unable to find desired frame (" << playback_frame_number + 1 << ") in frame buffer [" << buffer_begin_frame_number << "," << buffer_end_frame_number << "]";
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
    LOG(debug) << "Unable to find valid frame in frame buffer";
    return TS<SyncedMemory>();
}
int FrameGrabberBuffered::GetFrameNumber()
{
    return playback_frame_number;
}

void FrameGrabberBuffered::Buffer()
{
    cv::cuda::Stream read_stream;
    rmt_SetCurrentThreadName("FrameGrabberThread");
    LOG(info) << "Starting buffer thread";
    while(!boost::this_thread::interruption_requested())
    {   
        try
        {
            TS<SyncedMemory> frame;
            {
                boost::mutex::scoped_lock gLock(grabber_mtx);
                frame = GetNextFrameImpl(read_stream);
            }            
            if (!frame.empty())
            {
                boost::mutex::scoped_lock bLock(buffer_mtx);

                // Waiting for the reading thread to catch up
                while(buffer_begin_frame_number + 5 > playback_frame_number && frame_buffer.size() == frame_buffer.capacity())
                {
                    LOG(trace) << "Frame buffer is full and playback frame (" << playback_frame_number << ") is too close to the beginning of the frame buffer (" << buffer_begin_frame_number << ")";
                    if(update_signal)
                        (*update_signal)();
                    frame_read_cv.wait_for(bLock, boost::chrono::milliseconds(10));
                }
                buffer_end_frame_number = frame.frame_number;
                if(frame_buffer.size())
                    buffer_begin_frame_number = frame_buffer[0].frame_number;
                frame_buffer.push_back(frame);
                frame_grabbed_cv.notify_all();
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

void FrameGrabberBuffered::LaunchBufferThread()
{
    LOG(info);
    StopBufferThread();
    buffer_thread = boost::thread(boost::bind(&FrameGrabberBuffered::Buffer, this));
}

void FrameGrabberBuffered::StopBufferThread()
{
    LOG(info);
    buffer_thread.interrupt();
    DOIF_LOG_FAIL(buffer_thread.joinable(), buffer_thread.join(), warning);
}
int FrameGrabberInfo::LoadTimeout() const
{
    return 1000;
}
int FrameGrabberInfo::GetObjectInfoType()
{
    return IObjectInfo::ObjectInfoType::frame_grabber;
}
void FrameGrabberBuffered::Init(bool firstInit)
{
    IFrameGrabber::Init(firstInit);
    if(firstInit)
    {
        updateParameter<int>("Frame buffer size", 50);
    }else
    {
        if (parent_stream)
        {
            _callback_connections.push_back(parent_stream->GetSignalManager()->connect<void()>("StartThreads", std::bind(&FrameGrabberBuffered::LaunchBufferThread, this), this));
            _callback_connections.push_back(parent_stream->GetSignalManager()->connect<void()>("StopThreads", std::bind(&FrameGrabberBuffered::StopBufferThread, this), this));
        }
        
        LaunchBufferThread();
    }
    frame_buffer.set_capacity(*getParameter<int>("Frame buffer size")->Data());
    updateParameterPtr<boost::circular_buffer<TS<SyncedMemory>>>("Frame buffer", &frame_buffer)->type = Parameters::Parameter::Output;
    
    boost::function<void(cv::cuda::Stream*)> f =[&](cv::cuda::Stream*)
    {
        boost::mutex::scoped_lock lock(buffer_mtx);
        frame_buffer.set_capacity(*getParameter<int>("Frame buffer size")->Data());
    };
    RegisterParameterCallback("Frame buffer size", f, true, true);
}

void FrameGrabberBuffered::Serialize(ISimpleSerializer* pSerializer)
{
    IFrameGrabber::Serialize(pSerializer);
}
