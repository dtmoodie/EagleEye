#include "ImageWriter.h"
#include <boost/filesystem.hpp>


using namespace aq;
using namespace aq::Nodes;

bool ImageWriter::ProcessImpl()
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
    if(frequency == 0 && request_write == false)
        return true;
    if (request_write || (frameSkip >= frequency) || frequency == -1)
    {
        request_write_param.UpdateData(false);
        std::stringstream ss;
        if(!boost::filesystem::exists(save_directory))
        {
            boost::filesystem::create_directories(save_directory);
        }
        ss << save_directory.string() << "/" << base_name << std::setfill('0') << std::setw(4) << frame_count << ext;
        ++frame_count;
        std::string save_name = ss.str();
        if(input_image->getSyncState() < SyncedMemory::DEVICE_UPDATED)
        {
            cv::imwrite(save_name, input_image->getMat(Stream()));
        }else
        {
            input_image->synchronize(Stream());
            cv::Mat mat = input_image->getMat(Stream());
            cuda::enqueue_callback_async([mat, save_name]()->void
            {
                cv::imwrite(save_name, mat);
            }, Stream());
        }
        frameSkip = 0;
    }
    ++frameSkip;
    return true;
}
void ImageWriter::snap()
{
    request_write_param.UpdateData(true);
}

MO_REGISTER_CLASS(ImageWriter)
