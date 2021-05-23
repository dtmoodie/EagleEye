#include "ImageWriter.h"
#include <Aquila/nodes/NodeInfo.hpp>

#include <opencv2/imgcodecs.hpp>

#include <boost/filesystem.hpp>

#include <iomanip>

using namespace aq;
using namespace aq::nodes;

bool ImageWriter::processImpl()
{
    std::string ext;
    switch ((Extensions)extension.getValue())
    {
    case jpg:
        ext = ".jpg";
        break;
    case png:
        ext = ".png";
        break;
    case tiff:
        ext = ".tif";
        break;
    case bmp:
        ext = ".bmp";
        break;
    default:
        ext = ".jpg";
        break;
    }

    ++frame_count;
    if (frequency == 0 && request_write == false)
        return true;
    if (request_write || (frameSkip >= frequency) || frequency == -1)
    {
        request_write = false;
        std::stringstream ss;
        if (!boost::filesystem::exists(save_directory))
        {
            boost::filesystem::create_directories(save_directory);
        }
        ss << save_directory.string() << "/" << base_name << std::setfill('0') << std::setw(4) << frame_count << ext;
        std::string save_name = ss.str();
        auto stream = this->getStream();
        bool synchronize = false;
        cv::Mat mat = input_image->getMat(stream.get(), &synchronize);
        if (synchronize)
        {
            stream->pushWork([mat, save_name](mo::IAsyncStream&) -> void { cv::imwrite(save_name, mat); });
        }
        else
        {
            cv::imwrite(save_name, mat);
        }
        frameSkip = 0;
    }
    ++frameSkip;
    return true;
}
void ImageWriter::snap()
{
    request_write = true;
}

MO_REGISTER_CLASS(ImageWriter)
