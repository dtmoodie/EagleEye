#pragma once
#include <aqframegrabbers_export.hpp>

#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/types/SyncedImage.hpp"


namespace aqframegrabbers
{
    class aqframegrabbers_EXPORT GrabberImage : public aq::nodes::IGrabber
    {
      public:
        static int canLoad(const std::string& path);
        static int loadTimeout();

        MO_DERIVE(GrabberImage, aq::nodes::IGrabber)
            SOURCE(aq::SyncedImage, output)
            OUTPUT(std::string, image_name)
        MO_END;

        bool loadData(const std::string& path) override;
        bool grab() override;

        cv::Mat image;
        size_t count = 0;
        std::string m_path;
    };
} // namespace aqframegrabbers
