#include "image.h"
#include <MetaObject/MetaObjectInfo.hpp>
#include <EagleLib/Nodes/FrameGrabberInfo.hpp>
#include "ObjectInterfacePerModule.h"
#include <opencv2/imgcodecs.hpp>
#include <boost/filesystem.hpp>

using namespace ::EagleLib;
using namespace ::EagleLib::Nodes;

bool frame_grabber_image::LoadFile(const ::std::string& file_path)
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

long long frame_grabber_image::GetFrameNumber()
{
    return 0;
}

long long frame_grabber_image::GetNumFrames()
{
    return 1;
}

::std::string frame_grabber_image::GetSourceFilename()
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
TS<SyncedMemory> frame_grabber_image::GetFrameRelative(int index, cv::cuda::Stream& stream)
{
    return GetFrame(0, stream);
}
rcc::shared_ptr<ICoordinateManager> frame_grabber_image::GetCoordinateManager()
{
    return coordinate_manager;
}

int frame_grabber_image::CanLoadDocument(const std::string& document) const
{
    auto path = boost::filesystem::path(document);
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".png" || ext == ".tif") ? 3 : 0;
}


MO_REGISTER_CLASS(frame_grabber_image);
