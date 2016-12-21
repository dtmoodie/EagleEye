#include "ImageWriter.h"



using namespace EagleLib;
using namespace EagleLib::Nodes;

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
    if (request_write || (frameSkip >= frequency) || frequency == -1)
    {
        request_write_param.UpdateData(false);
        std::stringstream ss;
        ss << save_directory << base_name << std::setfill('0') << std::setw(4) << frame_count << ext;
        ++frame_count;
        std::string save_name = ss.str();
        if(input_image->GetSyncState() < SyncedMemory::DEVICE_UPDATED)
        {
            cv::imwrite(save_name, input_image->GetMat(Stream()));
        }else
        {
            input_image->Synchronize(Stream());
            cv::Mat mat = input_image->GetMat(Stream());
            cuda::enqueue_callback_async([mat, save_name]()->void
            {
                cv::imwrite(save_name, mat);
            }, Stream());
        }
    }
    ++frameSkip;
    return true;
}
MO_REGISTER_CLASS(ImageWriter)
