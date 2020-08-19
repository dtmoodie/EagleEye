#include <Aquila/types/SyncedMemory.hpp>

#include "Aquila/framegrabbers/GrabberInfo.hpp"
#include "image.h"
#include "precompiled.hpp"
#include <opencv2/imgcodecs.hpp>

using namespace aq;
using namespace aq::nodes;

bool GrabberImage::loadData(const std::string& path)
{
    image = cv::imread(path);
    if (!image.empty())
    {
        ++count;
        image_name_param.publish(path, mo::tag::_frame_number = count, mo::tag::_timestamp = mo::ms * (33 * count));
        output_param.publish(image, mo::tag::_frame_number = count, mo::tag::_timestamp = mo::ms * (33 * count));
        return true;
    }
    return false;
}

bool GrabberImage::grab()
{
    if (!image.empty())
    {
        output_param.publish(image, mo::tag::_frame_number = count, mo::tag::_timestamp = mo::ms * (33 * count));
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

int GrabberImage::loadTimeout()
{
    return 5000;
}

MO_REGISTER_CLASS(GrabberImage);
