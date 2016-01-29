#include "rtsp.h"
#include "ObjectInterfacePerModule.h"
using namespace EagleLib;

std::string frame_grabber_rtsp::frame_grabber_rtsp_info::GetObjectName()
{
    return "frame_grabber_rtsp";
}
std::string frame_grabber_rtsp::frame_grabber_rtsp_info::GetObjectTooltip()
{
    return "";
}
std::string frame_grabber_rtsp::frame_grabber_rtsp_info::GetObjectHelp()
{
    return "";
}
bool frame_grabber_rtsp::frame_grabber_rtsp_info::CanLoadDocument(const std::string& document) const
{
    std::string rtsp("rtsp");
    if(document.compare(0, rtsp.length(), rtsp) == 0)
    {
        return true;
    }
    return false;
}
int frame_grabber_rtsp::frame_grabber_rtsp_info::LoadTimeout() const
{
    return 3000;
}

int frame_grabber_rtsp::frame_grabber_rtsp_info::Priority() const
{
    return 9;
}


bool frame_grabber_rtsp::LoadFile(const std::string& file_path)
{
    //rtspsrc location=rtsp://root:12369pp@192.168.1.52:554/axis-media/media.amp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink
    std::string gstreamer_string = "rtspsrc location=" + file_path + " ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, width=1920, height=1080 ! appsink";
    return frame_grabber_cv::h_LoadFile(gstreamer_string);
}

shared_ptr<ICoordinateManager> frame_grabber_rtsp::GetCoordinateManager()
{
    return coordinate_manager;
}
static frame_grabber_rtsp::frame_grabber_rtsp_info info;
REGISTERCLASS(frame_grabber_rtsp, &info);