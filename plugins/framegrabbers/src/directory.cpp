#include "directory.h"
#include "precompiled.hpp"
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>
#include <algorithm>

using namespace aq;
using namespace aq::nodes;

int FrameGrabberDirectory::canLoadPath(const std::string& doc)
{
    int max_priority = 0;
    if(boost::filesystem::is_directory(doc))
    {
        auto constructors = mo::MetaObjectFactory::instance()->
                getConstructors(aq::nodes::IGrabber::getHash());
        std::vector<GrabberInfo*> infos;
        for(auto ctr : constructors)
        {
            if(auto fginfo = dynamic_cast<GrabberInfo*>(ctr->GetObjectInfo()))
            {
                infos.push_back(fginfo);
            }
        }
        boost::filesystem::directory_iterator end_itr;
        for (boost::filesystem::directory_iterator itr(doc); itr != end_itr; ++itr)
        {
            auto path = itr->path();
            if (is_regular_file(path))
            {
                for(auto info : infos)
                    max_priority = std::max(max_priority, info->canLoad(path.string()));
            }
        }
    }
    if(max_priority > 0)
        return max_priority + 1;
    return 0;
}

bool FrameGrabberDirectory::processImpl()
{
    if(fg)
    {
        if(frame_index < files_on_disk.size())
        {
            if(fg->loadData(files_on_disk[frame_index]))
            {
                ++frame_index;
                if(frame_index >= files_on_disk.size())
                {
                    sig_eos();
                }
            }
        }
    }
    return true;
}

void FrameGrabberDirectory::restart()
{
    frame_index = 0;
}

bool FrameGrabberDirectory::loadData(std::string file_path)
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
                extension_map[itr->path().extension().string()].push_back(itr->path().string());
            }
        }
        for(auto& itr: extension_map)
        {
            std::sort(itr.second.begin(), itr.second.end());
        }
        auto constructors = mo::MetaObjectFactory::instance()->
                getConstructors(aq::nodes::IGrabber::getHash());
        std::vector<int> load_count(constructors.size(), 0);
        std::vector<int> priorities(constructors.size(), 0);
        for(int i = 0; i < constructors.size(); ++i)
        {
            if(auto info = dynamic_cast<GrabberInfo*>(constructors[i]->GetObjectInfo()))
            {
                for(auto& file : extension_map)
                {
                    int priority = info->canLoad(file.second[0]);
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
            auto info = dynamic_cast<GrabberInfo*>(constructors[idx]->GetObjectInfo());
            for(auto& file : extension_map)
            {
                if(info->canLoad(file.second[0]))
                {
                    files_on_disk.insert(files_on_disk.end(), file.second.begin(), file.second.end());
                }
            }
            this->addComponent(fg);
            return files_on_disk.size() != 0;
        }
    }
    return false;
}

MO_REGISTER_CLASS(FrameGrabberDirectory);

