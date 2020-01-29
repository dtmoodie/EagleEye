#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "aqframegrabbers/aqframegrabbers_export.hpp"
namespace aq
{
namespace nodes
{

class aqframegrabbers_EXPORT FrameGrabberDirectory : public IFrameGrabber
{
  public:
    static int canLoadPath(const std::string& doc);
    static int loadTimeout();

    MO_DERIVE(FrameGrabberDirectory, IFrameGrabber)
        STATUS(int, frame_index, 0)
        MO_SIGNAL(void, eos)
        MO_SIGNAL(void, update)
        PARAM(bool, synchronous, false)
        MO_SLOT(void, nextFrame)
        MO_SLOT(void, prevFrame)
    MO_END

    virtual bool loadData(const std::string file_path) override;
    virtual void restart() override;

  protected:
    virtual bool processImpl() override;

  private:
    std::string loaded_file;
    std::vector<std::string> files_on_disk;
    rcc::shared_ptr<IGrabber> fg; // internal frame grabber used for loading the actual files
    bool step = false;
};
}
}
