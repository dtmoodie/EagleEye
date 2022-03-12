#include <ct/types/opencv.hpp>

#include <Aquila/types/SyncedMemory.hpp>

#include "Aquila/framegrabbers/GrabberInfo.hpp"
#include "image.h"
#include "precompiled.hpp"
#include <opencv2/imgcodecs.hpp>

using namespace aq;
using namespace aq::nodes;

namespace aqframegrabbers
{
    bool GrabberImage::loadData(const std::string& path)
    {
        image = cv::imread(path);
        if (!image.empty())
        {
            auto stream = this->getStream();
            image_name.publish(path, mo::tags::fn = count);
            output.publish(aq::SyncedImage(image, PixelFormat::kBGR, stream), mo::tags::fn = count);
            if(m_path != path)
            {
                ++count;
                m_path = path;
            }
            return true;
        }
        return false;
    }

    bool GrabberImage::grab()
    {
        if (!image.empty())
        {
            // published on load, no timestamp for images
            // output.publish(image, mo::tags::fn = 0, mo::tags::timestamp = mo::ms * 0);
            return true;
        }
        return false;
    }

    int GrabberImage::canLoad(const std::string& document)
    {
        auto path = boost::filesystem::path(document);
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return (ext == ".jpg" || ext == ".png" || ext == ".tif") ? 3 : 0;
    }

    int GrabberImage::loadTimeout() { return 5000; }
} // namespace aqframegrabbers

using namespace aqframegrabbers;
MO_REGISTER_CLASS(GrabberImage);

