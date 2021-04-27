#ifndef AQGSTREAMER_IMAGE_GRABBER_HPP
#define AQGSTREAMER_IMAGE_GRABBER_HPP
#include "gstreamer.hpp"

#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "aqgstreamer/aqgstreamer_export.hpp"

namespace aq
{
namespace grabbers
{
    class aqgstreamer_EXPORT GstreamerImageGrabber : virtual public aqgstreamer::GstreamerSrcBase,
                                                     virtual public nodes::IGrabber
    {
      public:
        static int canLoad(const std::string& doc);
        static int loadTimeout();

        MO_DERIVE(GstreamerImageGrabber, IGrabber)
            SOURCE(SyncedImage, image)
            MO_SIGNAL(void, update)
            PARAM(bool, use_system_time, false)
        MO_END;

        void initCustom(bool first_init) override;

      protected:
        GstFlowReturn onPull() override;
        bool loadData(const ::std::string& file_path) override;
        bool grab() override;
    };
} // namespace grabbers
}

#endif // AQGSTREAMER_IMAGE_GRABBER_HPP
