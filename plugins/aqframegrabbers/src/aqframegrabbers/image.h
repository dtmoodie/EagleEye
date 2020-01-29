#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include "aqframegrabbers/aqframegrabbers_export.hpp"

namespace aq
{
namespace nodes
{
class aqframegrabbers_EXPORT GrabberImage : public IGrabber
{
  public:
    static int canLoad(const std::string& path);
    static int loadTimeout();
    MO_DERIVE(GrabberImage, IGrabber)
        SOURCE(SyncedMemory, output, {})
        OUTPUT(std::string, image_name, {})
    MO_END

    virtual bool loadData(const std::string& path) override;
    virtual bool grab() override;
    cv::Mat image;
    size_t count = 0;
};
}
}
