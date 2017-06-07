#include "image.h"
#include "precompiled.hpp"
#include <opencv2/imgcodecs.hpp>
#include "Aquila/framegrabbers/GrabberInfo.hpp"


using namespace aq;
using namespace aq::Nodes;

bool GrabberImage::loadData(const std::string& path)
{
    image = cv::imread(path);
    if(!image.empty())
    {
        output_param.updateData(image);
        return true;
    }
    return false;
}
bool GrabberImage::grab()
{
    if(!image.empty())
    {
        output_param.updateData(image);
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
