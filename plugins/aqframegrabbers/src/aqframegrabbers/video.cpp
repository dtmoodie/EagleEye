#include <Aquila/types/SyncedMemory.hpp>

#include "Aquila/framegrabbers/FrameGrabberInfo.hpp"
#include "boost/filesystem.hpp"
#include "video.h"
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
using namespace aq;
using namespace aq::nodes;
/*frame_grabber_video::~frame_grabber_video()
{
    StopThreads();
}
int frame_grabber_video::CanLoadDocument(const std::string& document)
{
    boost::filesystem::path path(document);
    auto extension = path.extension().string();
    return (extension == ".avi" || extension == ".mp4") ? 1 : 0;
}



MO_REGISTER_CLASS(frame_grabber_video);
*/