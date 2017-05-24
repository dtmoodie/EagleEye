#include "image.h"
#include "precompiled.hpp"
#include <opencv2/imgcodecs.hpp>
#include "Aquila/Nodes/GrabberInfo.hpp"


using namespace aq;
using namespace aq::Nodes;

bool GrabberImage::Load(const std::string& path)
{
    image = cv::imread(path);
    if(!image.empty())
    {
        output_param.updateData(image);
        return true;
    }
    return false;
}
bool GrabberImage::Grab()
{
    if(!image.empty())
    {
        output_param.updateData(image);
        return true;
    }
    return false;
}

int GrabberImage::CanLoad(const std::string& document)
{
    auto path = boost::filesystem::path(document);
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".png" || ext == ".tif") ? 3 : 0;
}

int GrabberImage::Timeout()
{
    return 5000;
}


MO_REGISTER_CLASS(GrabberImage);
