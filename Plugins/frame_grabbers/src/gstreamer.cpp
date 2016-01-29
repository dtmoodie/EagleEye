#include "gstreamer.h"
#include "ObjectInterfacePerModule.h"
using namespace EagleLib;

std::string frame_grabber_gstreamer::frame_grabber_gstreamer_info::GetObjectName()
{
    return "frame_grabber_gstreamer";
}
std::string frame_grabber_gstreamer::frame_grabber_gstreamer_info::GetObjectTooltip()
{
    return "";
}
std::string frame_grabber_gstreamer::frame_grabber_gstreamer_info::GetObjectHelp()
{
    return "";
}
bool frame_grabber_gstreamer::frame_grabber_gstreamer_info::CanLoadDocument(const std::string& document) const
{
    return true;
}
int frame_grabber_gstreamer::frame_grabber_gstreamer_info::Priority() const
{
    return 2;
}


frame_grabber_gstreamer::frame_grabber_gstreamer()
{

}

bool frame_grabber_gstreamer::LoadFile(const std::string& file_path)
{
    return frame_grabber_cv::h_LoadFile(file_path);
}

shared_ptr<ICoordinateManager> frame_grabber_gstreamer::GetCoordinateManager()
{
    return coordinate_manager;
}

static frame_grabber_gstreamer::frame_grabber_gstreamer_info info;

REGISTERCLASS(frame_grabber_gstreamer, &info);