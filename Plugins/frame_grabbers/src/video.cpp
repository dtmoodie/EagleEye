#include "video.h"
#include "boost/filesystem.hpp"
#include <ObjectInterfacePerModule.h>
using namespace EagleLib;

std::string frame_grabber_video::frame_grabber_video_info::GetObjectName()
{
    return "frame_grabber_video";
}

std::string frame_grabber_video::frame_grabber_video_info::GetObjectTooltip()
{
    return "";
}

std::string frame_grabber_video::frame_grabber_video_info::GetObjectHelp()
{
    return "";
}

int frame_grabber_video::frame_grabber_video_info::CanLoadDocument(const std::string& document) const
{
    boost::filesystem::path path(document);
    auto extension = path.extension().string();
    return (extension == ".avi" || extension == ".mp4") ? 1 : 0;
}

int frame_grabber_video::frame_grabber_video_info::Priority() const
{
    return 1;
}

rcc::shared_ptr<ICoordinateManager> frame_grabber_video::GetCoordinateManager()
{
    return coordinate_manager;
}

static frame_grabber_video::frame_grabber_video_info info;

REGISTERCLASS(frame_grabber_video, &info);