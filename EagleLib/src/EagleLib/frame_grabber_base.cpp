#include "frame_grabber_base.h"
#include "EagleLib/Logging.h"
#include "EagleLib/DataStreamManager.h"
#include "Remotery.h"

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
        update_signal = stream->GetSignalManager()->GetSignal<void()>("update", this, stream->get_stream_id());
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
        update_signal = parent_stream->GetSignalManager()->GetSignal<void()>("update", this, parent_stream->get_stream_id());
    }
}
FrameGrabberBuffered::FrameGrabberBuffered()
{
    buffer_frame_number = 0;
    playback_frame_number = 1;
    updateParameter<int>("Frame buffer size", 50);
    frame_buffer.set_capacity(50);
    
    boost::function<void(cv::cuda::Stream*)> f =[&](cv::cuda::Stream*)
    {
        frame_buffer.set_capacity(*getParameter<int>("Frame buffer size")->Data());
    };
    RegisterParameterCallback("Frame buffer size", f, true, true);
}
void FrameGrabberBuffered::InitializeFrameGrabber(DataStream* stream)
{
    IFrameGrabber::InitializeFrameGrabber(stream);
    if (stream)
    {
        connections.push_back(stream->GetSignalManager()->Connect<void()>("StartThreads", std::bind(&FrameGrabberBuffered::LaunchBufferThread, this), this, -1));
        connections.push_back(stream->GetSignalManager()->Connect<void()>("StopThreads", std::bind(&FrameGrabberBuffered::StopBufferThread, this), this, -1));
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
        boost::this_thread::interruptible_wait(10);
    }
    for(auto& itr: frame_buffer)
    {
        if(itr.frame_number == index)
        {
            return itr;
        }
    }
    auto frame = GetFrameImpl(index, stream);
    frame_buffer.push_back(frame);
    return frame;
}

TS<SyncedMemory> FrameGrabberBuffered::GetNextFrame(cv::cuda::Stream& stream)
{
    playback_frame_number++;
    (*update_signal)();
    return GetFrame(playback_frame_number, stream);
}
int FrameGrabberBuffered::GetFrameNumber()
{
    return playback_frame_number;
}

void FrameGrabberBuffered::Buffer()
{
    cv::cuda::Stream read_stream;
    rmt_SetCurrentThreadName("FrameGrabberThread");
    while(!boost::this_thread::interruption_requested())
    {
        if(frame_buffer.size() != frame_buffer.capacity() || buffer_frame_number == -1 || playback_frame_number > buffer_frame_number - frame_buffer.capacity() / 2)
        {
            auto frame = GetNextFrameImpl(read_stream);
            if(!frame.empty())
            {
                buffer_frame_number = frame.frame_number;
                if(update_signal)
                    (*update_signal)();
                boost::mutex::scoped_lock lock(buffer_mtx);
                frame_buffer.push_back(frame);
            }
        }
    }
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
    buffer_thread.join();
}
int FrameGrabberInfo::LoadTimeout() const
{
    return 100;
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
        
    }else
    {
        if (parent_stream)
        {
            connections.push_back(parent_stream->GetSignalManager()->Connect<void()>("StartThreads", std::bind(&FrameGrabberBuffered::LaunchBufferThread, this), this, -1));
            connections.push_back(parent_stream->GetSignalManager()->Connect<void()>("StopThreads", std::bind(&FrameGrabberBuffered::StopBufferThread, this), this, -1));
        }
        buffer_frame_number = -1;
        LaunchBufferThread();
    }
    updateParameterPtr<boost::circular_buffer<TS<SyncedMemory>>>("Frame buffer", &frame_buffer)->type = Parameters::Parameter::Output;
}

void FrameGrabberBuffered::Serialize(ISimpleSerializer* pSerializer)
{
    IFrameGrabber::Serialize(pSerializer);
    SERIALIZE(playback_frame_number);
    
}