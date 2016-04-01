#include "ProcessFuture.h"
#include "EagleLib/DataStreamManager.h"

using namespace EagleLib;
using namespace EagleLib::Nodes;

ProcessFuture::ProcessFuture()
{

}

ProcessFuture::~ProcessFuture()
{
    _thread.interrupt();
    _thread.join();
}

void ProcessFuture::Init(bool firstInit)
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

void ProcessFuture::process_ahead()
{
    std::unique_lock<std::recursive_mutex> lock(mtx);
    auto frame_grabber = GetDataStream()->GetFrameGrabber();
    
    if(!frame_grabber)
    {
        LOG(debug) << "No valid frame grabber";
        return;
    }
    cv::cuda::Stream stream;
    TS<SyncedMemory> frame;
    while(!boost::this_thread::interruption_requested())
    {
        int num_frames_ahead = *getParameter<int>("Num Frames Ahead")->Data();
        while(frame.empty())
        {
            _cv.wait_for(lock, std::chrono::milliseconds(100));
            frame = frame_grabber->GetFrameRelative(num_frames_ahead,stream);
        }
        // Process all children
        process(frame, stream);
    }
}