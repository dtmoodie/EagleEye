#include "image.h"
#include <boost/filesystem.hpp>
#include "ObjectInterfacePerModule.h"
#include <opencv2/imgcodecs.hpp>

using namespace EagleLib;

bool frame_grabber_image::LoadFile(const std::string& file_path)
{
    h_image = cv::imread(file_path);
    if(!h_image.empty())
    {
        d_image.upload(h_image);
        loaded_file = file_path;
        return true;
    }
    return false;
}

int frame_grabber_image::GetFrameNumber()
{
    return 0;
}
int frame_grabber_image::GetNumFrames()
{
    return 1;
}
std::string frame_grabber_image::GetSourceFilename()
{
    return loaded_file;
}

TS<SyncedMemory> frame_grabber_image::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return GetFrame(0, stream);
}
TS<SyncedMemory> frame_grabber_image::GetFrame(int index, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat d_out;
    cv::Mat h_out;
    d_image.copyTo(d_out, stream);
    h_image.copyTo(h_out);
    return TS<SyncedMemory>(0.0, 0, h_out, d_out);
}
TS<SyncedMemory> frame_grabber_image::GetNextFrame(cv::cuda::Stream& stream)
{
    return GetFrame(0, stream);
}

shared_ptr<ICoordinateManager> frame_grabber_image::GetCoordinateManager()
{
    return coordinate_manager;
}

std::string frame_grabber_image::frame_grabber_image_info::GetObjectName()
{
    return "frame_grabber_image";
}

std::string frame_grabber_image::frame_grabber_image_info::GetObjectTooltip()
{
    return "";
}

std::string frame_grabber_image::frame_grabber_image_info::GetObjectHelp()
{
    return "Frame grabber for static image files";
}

bool frame_grabber_image::frame_grabber_image_info::CanLoadDocument(const std::string& document) const
{
    auto path = boost::filesystem::path(document);
    auto ext = path.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".png" || ext == ".tif";
}
int frame_grabber_image::frame_grabber_image_info::Priority() const
{
    return 3;
}
static frame_grabber_image::frame_grabber_image_info info;
REGISTERCLASS(frame_grabber_image, &info);
