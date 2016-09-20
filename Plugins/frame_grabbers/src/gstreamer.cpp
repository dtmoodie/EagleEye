#include "gstreamer.h"
#include "ObjectInterfacePerModule.h"
#include "EagleLib/Nodes/FrameGrabberInfo.hpp"
#include <MetaObject/MetaObjectInfo.hpp>

#include <boost/filesystem.hpp>
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>

using namespace EagleLib;
using namespace EagleLib::Nodes;


int frame_grabber_gstreamer::CanLoadDocument(const std::string& document) const
{
    boost::filesystem::path path(document);
    // oooor a gstreamer pipeline.... 
    std::string appsink = "appsink";
    if(document.find(appsink) != std::string::npos)
        return 9;
    if(boost::filesystem::is_regular_file(path))
        return 2;
    return 0;
}



frame_grabber_gstreamer::frame_grabber_gstreamer():
    frame_grabber_cv()
{
    if (!gst_is_initialized())
    {
        char** argv;
        argv = new char*{ "-vvv" };
        int argc = 1;
        gst_init(&argc, &argv);
    }
}

bool frame_grabber_gstreamer::LoadFile(const std::string& file_path)
{
    return frame_grabber_cv::h_LoadFile(file_path);
}

rcc::shared_ptr<ICoordinateManager> frame_grabber_gstreamer::GetCoordinateManager()
{
    return coordinate_manager;
}



MO_REGISTER_CLASS(frame_grabber_gstreamer, &info);