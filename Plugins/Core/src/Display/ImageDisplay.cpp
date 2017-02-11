#include "ImageDisplay.h"

#include <MetaObject/Thread/InterThread.hpp>
#include <EagleLib/utilities/CudaCallbacks.hpp>
#include <EagleLib/utilities/UiCallbackHandlers.h>
#include <MetaObject/Logging/Profiling.hpp>



using namespace EagleLib;
using namespace EagleLib::Nodes;

bool QtImageDisplay::ProcessImpl()
{
    cv::Mat mat;
    bool stream_desynced = false;
    long long ts = -1;
    bool overlay = overlay_timestamp;
    if(image && !image->empty())
    {
        if(image->GetSyncState() == EagleLib::SyncedMemory::DEVICE_UPDATED)
        {
            stream_desynced = true;
        }
        mat = image->GetMat(Stream());
        ts = image_param.GetTimestamp();
    }
    if(cpu_mat)
    {
        mat = *cpu_mat;
    }
    
    std::string name = GetTreeName();
    if(!mat.empty())
    {
        size_t gui_thread_id = mo::ThreadRegistry::Instance()->GetThread(mo::ThreadRegistry::GUI);
        if(stream_desynced)
        {
            EagleLib::cuda::enqueue_callback_async(
                [mat, name, overlay, ts]()->void
            {
                PROFILE_RANGE(imshow);
                if(overlay)
                {
                    std::stringstream ss;
                    ss << "Timestamp: " << ts;
                    cv::putText(mat, ss.str(), cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0,255,0));
                }

                cv::imshow(name, mat);
            }, gui_thread_id, Stream());
        }else
        {
            mo::ThreadSpecificQueue::Push(
                        [name, mat, overlay, ts]()
            {
                PROFILE_RANGE(imshow);
                if(overlay)
                {
                    std::stringstream ss;
                    ss << "Timestamp: " << ts;
                    cv::putText(mat, ss.str(), cv::Point(20,40), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0,255,0));
                }
                cv::imshow(name, mat);
            }, gui_thread_id, this);
        }
        return true;
    }
    return false;
}

bool KeyPointDisplay::ProcessImpl()
{
    return true;
}

bool FlowVectorDisplay::ProcessImpl()
{
    return true;
}

bool HistogramDisplay::ProcessImpl()
{
    return true;
}

bool DetectionDisplay::ProcessImpl()
{
    return true;
}

bool OGLImageDisplay::ProcessImpl()
{
    return true;
}


MO_REGISTER_CLASS(QtImageDisplay);
MO_REGISTER_CLASS(KeyPointDisplay);
MO_REGISTER_CLASS(FlowVectorDisplay);
MO_REGISTER_CLASS(HistogramDisplay);
MO_REGISTER_CLASS(DetectionDisplay);
MO_REGISTER_CLASS(OGLImageDisplay);


