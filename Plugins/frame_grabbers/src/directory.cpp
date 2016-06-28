#include "directory.h"
#include <boost/filesystem.hpp>
#include "ObjectInterfacePerModule.h"
#include <EagleLib/rcc/external_includes/cv_imgcodec.hpp>
#include <boost/filesystem.hpp>
#include <parameters/ParameteredObjectImpl.hpp>
using namespace EagleLib;
frame_grabber_directory::frame_grabber_directory()
{
    frame_index = 0;
    

}
void frame_grabber_directory::NodeInit(bool firstInit)
{
    updateParameter("Frame Index", 0);
    updateParameter("Loaded file", std::string(""));
    updateParameter<boost::function<void(void)>>("Restart", [this]()->void
    {
        frame_index = 0;
        updateParameter("Frame Index", 0);
    });
}
void frame_grabber_directory::Serialize(ISimpleSerializer* pSerializer)
{
    IFrameGrabber::Serialize(pSerializer);
    SERIALIZE(frame_index);
    SERIALIZE(files_on_disk);
    SERIALIZE(loaded_images);
}
bool frame_grabber_directory::LoadFile(const std::string& file_path)
{
    boost::filesystem::path path(file_path);
    boost::filesystem::directory_iterator end_itr;
    if(boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
    {
        for(boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr)
        {
            if(boost::filesystem::is_regular_file(itr->status()))
            {
                files_on_disk.push_back(itr->path().string());
            }
        }
        return true;
    }
    return false;
}

int frame_grabber_directory::GetFrameNumber()
{
    return frame_index;
}
int frame_grabber_directory::GetNumFrames()
{
    return files_on_disk.size();
}
std::string frame_grabber_directory::GetSourceFilename()
{
    return files_on_disk[frame_index];
}

TS<SyncedMemory> frame_grabber_directory::GetCurrentFrame(cv::cuda::Stream& stream)
{
    return GetFrame(frame_index, stream);
}
TS<SyncedMemory> frame_grabber_directory::GetFrame(int index, cv::cuda::Stream& stream)
{
    // First check if this has already been loaded in the frame buffer
    if(files_on_disk.empty())
        return TS<SyncedMemory>(0.0, 0);
    if(index >= files_on_disk.size())
        index = files_on_disk.size() - 1;
    std::string file_name = files_on_disk[index];
    if (index != *getParameter<int>("Frame Index")->Data())
    {
        updateParameter("Frame Index", index);
        updateParameter("Loaded file", file_name);
    }
    
    for(auto& itr : loaded_images)
    {
        if(std::get<0>(itr) == file_name)
        {
            return TS<SyncedMemory>(0.0, 0, std::get<1>(itr), std::get<2>(itr));
        }
    }
    cv::Mat h_out = cv::imread(file_name);
    if(h_out.empty())
        return TS<SyncedMemory>(0.0, 0);
    cv::cuda::GpuMat d_mat;
    d_mat.upload(h_out, stream);
    loaded_images.push_back(std::make_tuple(file_name, h_out, d_mat));
    return TS<SyncedMemory>(0.0, 0, h_out, d_mat);
}
TS<SyncedMemory> frame_grabber_directory::GetNextFrame(cv::cuda::Stream& stream)
{
    if (frame_index == files_on_disk.size() - 1)
        return GetCurrentFrame(stream);
    return GetFrame(frame_index++, stream);
}

TS<SyncedMemory> frame_grabber_directory::GetFrameRelative(int index, cv::cuda::Stream& stream)
{
    index = frame_index + index;
    index = std::max((int)files_on_disk.size() - 1, index);
    index = std::min(0, index);
    return GetFrame(index, stream);
}

rcc::shared_ptr<ICoordinateManager> frame_grabber_directory::GetCoordinateManager()
{
    return coordinate_manager;
}

std::string frame_grabber_directory::frame_grabber_directory_info::GetObjectName()
{
    return "frame_grabber_directory";
}

std::string frame_grabber_directory::frame_grabber_directory_info::GetObjectTooltip()
{
    return "";
}

std::string frame_grabber_directory::frame_grabber_directory_info::GetObjectHelp()
{
    return "Frame grabber for static image files in a directory";
}

int frame_grabber_directory::frame_grabber_directory_info::CanLoadDocument(const std::string& document) const
{
    auto path = boost::filesystem::path(document);
    
    return (boost::filesystem::exists(path) && boost::filesystem::is_directory(path)) ? 1 : 0;
}
int frame_grabber_directory::frame_grabber_directory_info::Priority() const
{
    return 1;
}
static frame_grabber_directory::frame_grabber_directory_info info;

REGISTERCLASS(frame_grabber_directory, &info);