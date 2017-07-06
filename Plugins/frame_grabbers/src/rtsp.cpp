#ifdef HAVE_GSTREAMER
#include "rtsp.h"
#include "precompiled.hpp"


using namespace aq;
using namespace aq::nodes;

int GrabberRTSP::canLoad(const std::string& document)
{
    std::string rtsp("rtsp");
    if (document.compare(0, rtsp.length(), rtsp) == 0)
    {
        return 10;
    }
    return 0;
}
int GrabberRTSP::loadTimeout()
{
    return 10000;
}
bool GrabberRTSP::Load(const std::string& file_path)
{
    //rtspsrc location=rtsp://root:12369pp@192.168.1.52:554/axis-media/media.amp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink
    std::string file_to_load;
    if (loaded_document.size() && !file_path.size())
        file_to_load = loaded_document;
    else
        file_to_load = file_path;
#ifdef JETSON
    std::string gstreamer_string = "rtspsrc location=" + file_path + " ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink";
#else
    std::string gstreamer_string = "rtspsrc location=" + file_to_load + " ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw ! appsink";
#endif

    h_cam.release();
    MO_LOG(info) << "Attemping to load " << file_to_load;
    MO_LOG(debug) << "Gstreamer string: " << gstreamer_string;
    try
    {
        h_cam.reset(new cv::VideoCapture());
        if (h_cam)
        {
            if (h_cam->open(gstreamer_string, cv::CAP_GSTREAMER))
            {
                loaded_document = file_to_load;
                
                return true;
            }
        }
    }
    catch (cv::Exception& e)
    {
    }
    return false;
}

/*
TS<SyncedMemory> frame_grabber_rtsp::GetNextFrameImpl(cv::cuda::Stream& stream)
{
    if(_reconnect)
        LoadFile();
    try
    {
        if (h_cam)
        {
            cv::Mat h_mat;
            if (h_cam->read(h_mat))
            {
                if (!h_mat.empty())
                {
                    cv::cuda::GpuMat d_mat;
                    d_mat.upload(h_mat, stream);
                    current_frame = TS<SyncedMemory>(mo::Time_t(mo::ms*h_cam->get(cv::CAP_PROP_POS_MSEC)), (size_t)frame_count++, h_mat, d_mat);
                    return TS<SyncedMemory>(current_frame.timestamp, current_frame.frame_number, current_frame.clone(stream));
                }
            }
        }
    }catch(cv::Exception &e)
    {

    }
    catch(...)
    {

    }
    _reconnect = true;
    return TS<SyncedMemory>();
}
void frame_grabber_rtsp::nodeInit(bool firstInit)
{
    frame_grabber_cv::Init(firstInit);
    _reconnect = false;
    UpdateParameter<std::function<void(void)>>("Reconnect", [this]()->void
    {
        this->_reconnect = true;
    });
}
bool frame_grabber_rtsp::LoadFile(const std::string& file_path)
{
    //rtspsrc location=rtsp://root:12369pp@192.168.1.52:554/axis-media/media.amp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink
    std::string file_to_load;
    if(loaded_document.size() && !file_path.size())
        file_to_load = loaded_document;
    else
        file_to_load = file_path;
#ifdef JETSON
    std::string gstreamer_string = "rtspsrc location=" + file_path + " ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink";
#else
    std::string gstreamer_string = "rtspsrc location=" + file_to_load + " ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw ! appsink";
#endif
    
    h_cam.release();
    LOG_NODE(info) << "Attemping to load " << file_to_load;
    LOG_NODE(debug) << "Gstreamer string: " << gstreamer_string;
    _reconnect = false;
    playback_frame_number = 0;
    try
    {
        h_cam.reset(new cv::VideoCapture());
        if (h_cam)
        {
            if (h_cam->open(gstreamer_string, cv::CAP_GSTREAMER))
            {
                loaded_document = file_to_load;
                playback_frame_number = h_cam->get(cv::CAP_PROP_POS_FRAMES) + 1;
                LOG_NODE(info) << "Load success, first frame number: " << playback_frame_number;
                frame_buffer.clear();
                buffer_begin_frame_number = playback_frame_number.load();
                buffer_end_frame_number = playback_frame_number.load();
                return true;
            }
        }
    }
    catch (cv::Exception& e)
    {
    }
    return false;
}

void frame_grabber_rtsp::seek_relative_msec(double msec)
{
    if(h_cam)
    {
        double current = h_cam->get(cv::CAP_PROP_POS_MSEC);
        if(!h_cam->set(cv::CAP_PROP_POS_MSEC, current + msec))
        {
            LOG_NODE(debug) << "Failed to seek by (" << msec << " ms) from " << current << " ms";
        }
    }
}

MO_REGISTER_CLASS(frame_grabber_rtsp);
*/
#endif
