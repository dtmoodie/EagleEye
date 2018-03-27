#pragma once
#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "aqgstreamer_export.hpp"
#include "gstreamer.hpp"

namespace aq
{
    namespace grabbers
    {
        class aqgstreamer_EXPORT GstreamerImageGrabber : virtual public gstreamer_src_base,
                                                         virtual public nodes::IGrabber
        {
          public:
            static int canLoad(const std::string& doc);
            static int loadTimeout() { return 10000; }
            MO_DERIVE(GstreamerImageGrabber, IGrabber)
                SOURCE(SyncedMemory, image, {})
                MO_SIGNAL(void, update)
            MO_END
          protected:
            virtual GstFlowReturn on_pull() override;
            virtual bool loadData(const ::std::string& file_path) override;
            virtual bool grab() override;
        };
    }
}
