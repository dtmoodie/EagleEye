#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "aqframegrabbers/aqframegrabbers_export.hpp"

namespace aqframegrabbers
{

    class aqframegrabbers_EXPORT FrameGrabberDirectory : public aq::nodes::IFrameGrabber
    {
      public:
        static int canLoadPath(const std::string& doc);
        static int loadTimeout();

        MO_DERIVE(FrameGrabberDirectory, aq::nodes::IFrameGrabber)
            STATUS(int, frame_index, 0)
            MO_SIGNAL(void, eos)
            MO_SIGNAL(void, update)
            PARAM(bool, synchronous, false)
            MO_SLOT(void, nextFrame)
            MO_SLOT(void, prevFrame)

        MO_END;

        bool loadData(const std::string file_path) override;
        void restart() override;

      protected:
        bool processImpl() override;

      private:
        std::string loaded_file;
        std::vector<std::string> files_on_disk;
        rcc::shared_ptr<aq::nodes::IGrabber> fg; // internal frame grabber used for loading the actual files
        bool step = false;
    };
} // namespace aqframegrabbers
