#include <ct/types/opencv.hpp>

#include <Aquila/types/SyncedMemory.hpp>

#include "boost/filesystem.hpp"
#include "video.h"
#include <Aquila/framegrabbers/GrabberInfo.hpp>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

namespace aqframegrabbers
{

    int FrameGrabberVideo::canLoad(const std::string& document)
    {
        boost::filesystem::path path(document);
        if (!boost::filesystem::exists(path))
        {
            return 0;
        }
        auto extension = path.extension().string();
        return (extension == ".avi" || extension == ".mp4" || extension == ".mkv") ? 1 : 0;
    }
} // namespace aqframegrabbers
using namespace aqframegrabbers;

MO_REGISTER_CLASS(FrameGrabberVideo);
