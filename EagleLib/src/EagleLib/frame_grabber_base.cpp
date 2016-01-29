#include "frame_grabber_base.h"

#include "EagleLib/DataStreamManager.h"
using namespace EagleLib;
IFrameGrabber::IFrameGrabber()
{
    update_signal = nullptr;
}
std::string IFrameGrabber::GetSourceFilename()
{
    return loaded_document;
}
void IFrameGrabber::InitializeFrameGrabber(DataStream* stream)
{
    if(stream)
    {
        update_signal = stream->GetSignalManager()->GetSignal<void()>("update", this, stream->get_stream_id());
    }
}
void IFrameGrabber::Serialize(ISimpleSerializer* pSerializer)
{
    ParameteredIObject::Serialize(pSerializer);
    SERIALIZE(update_signal);
    SERIALIZE(loaded_document);
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
    if(!buffer_thread.joinable())
    {
        LaunchBufferThread();
    }
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
    while(!boost::this_thread::interruption_requested())
    {
        if(frame_buffer.size() != frame_buffer.capacity() || playback_frame_number > buffer_frame_number - frame_buffer.capacity() / 2)
        {
            auto frame = GetNextFrameImpl(read_stream);
            {
                buffer_frame_number = frame.frame_number;
                boost::mutex::scoped_lock lock(buffer_mtx);
                frame_buffer.push_back(frame);
            }
        }
    }
}

void FrameGrabberBuffered::LaunchBufferThread()
{
    StopBufferThread();
    buffer_thread = boost::thread(boost::bind(&FrameGrabberBuffered::Buffer, this));
}

void FrameGrabberBuffered::StopBufferThread()
{
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
    if(firstInit)
    {
        
    }else
    {
        LaunchBufferThread();
    }
    updateParameterPtr<boost::circular_buffer<TS<SyncedMemory>>>("Frame buffer", &frame_buffer)->type = Parameters::Parameter::Output;
}

void FrameGrabberBuffered::Serialize(ISimpleSerializer* pSerializer)
{
    IFrameGrabber::Serialize(pSerializer);
    SERIALIZE(playback_frame_number);
    SERIALIZE(buffer_frame_number);
}