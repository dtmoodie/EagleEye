#include "directory.h"
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>
#include "precompiled.hpp"

using namespace aq;
using namespace aq::Nodes;

frame_grabber_directory::frame_grabber_directory()
{
    frame_index = 0;
}
bool frame_grabber_directory::ProcessImpl()
{
    auto frame = GetNextFrame(Stream());
    if(!frame.empty())
    {
        this->current_frame_param.UpdateData(frame, frame.frame_number, _ctx);
        return true;
    }
    return false;
}
void frame_grabber_directory::Restart()
{

}

void frame_grabber_directory::NodeInit(bool firstInit)
{
    //updateParameter("Frame Index", 0);
    //updateParameter("Loaded file", std::string(""));
    /*updateParameter<boost::function<void(void)>>("Restart", [this]()->void
    {
        frame_index = 0;
        updateParameter("Frame Index", 0);
    });*/
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
    auto path = boost::filesystem::path(file_path);
    if(boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
    {
        boost::filesystem::directory_iterator end_itr;
        std::vector<std::string> files;
        // cycle through the directory
        std::map<std::string, std::vector<std::string>> extension_map;
        for (boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr)
        {
            if (is_regular_file(itr->path())) 
            {
                //files.push_back(itr->path().string());
                extension_map[itr->path().extension().string()].push_back(itr->path().string());
            }
        }
        for(auto& itr: extension_map)
        {
            std::sort(itr.second.begin(), itr.second.end());
        }
        auto constructors = mo::MetaObjectFactory::Instance()->
                GetConstructors(aq::Nodes::IFrameGrabber::s_interfaceID);
        std::vector<int> load_count(constructors.size(), 0);
        std::vector<int> priorities(constructors.size(), 0);
        for(int i = 0; i < constructors.size(); ++i)
        {
            if(auto info = dynamic_cast<FrameGrabberInfo*>(constructors[i]->GetObjectInfo()))
            {
                for(auto& file : extension_map)
                {
                    int priority = info->CanLoadDocument(file.second[0]);
                    if(priority > 0)
                    {
                        ++load_count[i];
                        priorities[i] = priority;
                    }
                }
            }
        }
        auto itr = std::max_element(load_count.begin(), load_count.end());
        if(itr != load_count.end() && (*itr) != 0)
        {
            long long idx = itr - load_count.begin();
            this->fg = constructors[idx]->Construct();
            this->fg->Init(true);
            auto info = dynamic_cast<FrameGrabberInfo*>(constructors[idx]->GetObjectInfo());
            for(auto& file : extension_map)
            {
                if(info->CanLoadDocument(file.second[0]))
                {
                    files_on_disk.insert(files_on_disk.end(), file.second.begin(), file.second.end());
                }
            }
            return files_on_disk.size() != 0;
        }
    }
    return false;
}

long long frame_grabber_directory::GetFrameNumber()
{
    return frame_index;
}
long long frame_grabber_directory::GetNumFrames()
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
    if(index == files_on_disk.size() - 1)
        sig_eos();
    if(files_on_disk.empty())
        return TS<SyncedMemory>(0.0, (long long)0);
    if(index >= files_on_disk.size())
        index = static_cast<int>(files_on_disk.size() - 1);
    std::string file_name = files_on_disk[index];

    cv::Mat h_out = cv::imread(file_name);

    if(h_out.empty())
        return TS<SyncedMemory>(0.0, (long long)0);

    this->_modified = true;

    return TS<SyncedMemory>(0.0, (long long)index, h_out);
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

int frame_grabber_directory::CanLoadDocument(const std::string& document)
{
    auto path = boost::filesystem::path(document);
    if (boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
    {
        std::map<std::string, std::vector<std::string>> extension_map;
        boost::filesystem::directory_iterator end_itr;
        std::vector<std::string> files;
        // cycle through the directory
        for (boost::filesystem::directory_iterator itr(path); itr != end_itr; ++itr)
        {
            if (is_regular_file(itr->path()))
            {
                extension_map[itr->path().extension().string()].push_back(itr->path().string());
            }
        }

        auto constructors = mo::MetaObjectFactory::Instance()->
                GetConstructors(aq::Nodes::IFrameGrabber::s_interfaceID);
        std::vector<int> load_count(constructors.size(), 0);
        std::vector<int> priorities(constructors.size(), 0);
        for (int i = 0; i < constructors.size(); ++i)
        {
            if (auto info = dynamic_cast<FrameGrabberInfo*>(constructors[i]->GetObjectInfo()))
            {
                for(auto& file : extension_map)
                {
                    int priority = info->CanLoadDocument(file.second[0]);
                    if (priority > 0)
                    {
                        ++load_count[i];
                        priorities[i] = priority;
                    }
                }
            }
        }
        auto itr = std::max_element(load_count.begin(), load_count.end());
        if (itr != load_count.end())
        {
            long long idx = itr - load_count.begin();
            return priorities[idx];
        }
    }
    return 0;
}


MO_REGISTER_CLASS(frame_grabber_directory);
