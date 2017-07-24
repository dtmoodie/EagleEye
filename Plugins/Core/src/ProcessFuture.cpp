/*#include "ProcessFuture.h"
#include "Aquila/DataStreamManager.h"
#include "Remotery.h"
#include <Aquila/frame_grabber_base.h>
#include "Aquila/Signals.h"
using namespace aq;
using namespace aq::nodes;

ProcessFuture::ProcessFuture()
{
    _pause = false;
}

ProcessFuture::~ProcessFuture()
{
    _thread.interrupt();
    _thread.join();
}

void ProcessFuture::nodeInit(bool firstInit)
{
    updateParameter("Num Frames Ahead", 5);
    _thread = boost::thread(boost::bind(&ProcessFuture::process_ahead, this));
}

TS<SyncedMemory> ProcessFuture::process(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    if(boost::this_thread::get_id() != _thread.get_id())
    {
        _cv.notify_all();
        return input;
    }
    return Node::process(input, stream);
}
TS<SyncedMemory> ProcessFuture::doProcess(TS<SyncedMemory>& input, cv::cuda::Stream& stream)
{
    return input;
}

void ProcessFuture::process_ahead()
{
    std::unique_lock<std::recursive_mutex> lock(mtx);
    auto frame_grabber = getDataStream()->GetFrameGrabber();
    rmt_SetCurrentThreadName("ProcessFuture_thread");
    if(!frame_grabber)
    {
        MO_LOG(debug) << "No valid frame grabber";
        return;
    }
    cv::cuda::Stream stream;
    
    while(!boost::this_thread::interruption_requested())
    {
        TS<SyncedMemory> frame;
        while (_pause)
        {
            _cv.wait(lock);
        }
        while(frame.empty())
        {
            int num_frames_ahead = *getParameter<int>("Num Frames Ahead")->Data();
            _cv.wait_for(lock, std::chrono::milliseconds(100));
            frame = frame_grabber->GetFrameRelative(num_frames_ahead,stream);
        }
        // Process all children
        process(frame, stream);
    }
}
void ProcessFuture::SetDataStream(IDataStream* stream)
{
    Node::SetDataStream(stream);
    _callback_connections.push_back(stream->GetSignalManager()->connect<void()>("StartThreads", std::bind(&ProcessFuture::start_thread, this), this));
    _callback_connections.push_back(stream->GetSignalManager()->connect<void()>("StopThreads", std::bind(&ProcessFuture::stop_thread, this), this));
}
void ProcessFuture::stop_thread()
{
    _pause = true;
}
void ProcessFuture::start_thread()
{
    _pause = false;
    _cv.notify_all();
}



REGISTERCLASS(ProcessFuture, &s_process_future_info)*/