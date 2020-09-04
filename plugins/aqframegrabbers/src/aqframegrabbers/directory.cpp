#include "directory.h"
#include "precompiled.hpp"
#include <Aquila/framegrabbers/FrameGrabberInfo.hpp>
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>
#include <algorithm>

namespace aqframegrabbers
{
    int FrameGrabberDirectory::loadTimeout() { return 10000; }

    int FrameGrabberDirectory::canLoadPath(const std::string& doc)
    {
        int max_priority = 0;
        if (!boost::filesystem::is_directory(doc))
        {
            MO_LOG(debug, "{} is not a directory", doc);
            return 0;
        }

        std::shared_ptr<mo::MetaObjectFactory> factory = mo::MetaObjectFactory::instance();
        const auto hash = aq::nodes::IGrabber::getHash();
        auto constructors = factory->getConstructors(hash);
        std::vector<const aq::nodes::GrabberInfo*> infos;
        for (auto ctr : constructors)
        {
            if (auto fginfo = dynamic_cast<const aq::nodes::GrabberInfo*>(ctr->GetObjectInfo()))
            {
                infos.push_back(fginfo);
            }
        }
        if (infos.size() == 0)
        {
            MO_LOG(debug, "Did not find any valid grabbers for directory contents");
            return 0;
        }

        boost::filesystem::directory_iterator end_itr;
        for (boost::filesystem::directory_iterator itr(doc); itr != end_itr; ++itr)
        {
            auto path = itr->path();
            std::string path_str = path.string();
            if (is_regular_file(path))
            {
                for (auto info : infos)
                {
                    auto p = info->canLoad(path_str);
                    MO_LOG(trace, "Priority for {} is {} from {}", path_str, p, info->getDisplayName());
                    max_priority = std::max(max_priority, p);
                }
            }
            else
            {
                MO_LOG(trace, "{} is not a regular file", path_str);
            }
        }
        if (max_priority > 0)
        {
            return max_priority + 1;
        }
        MO_LOG(debug, "FrameGrabberDirectory cannot load {}", doc);
        return 0;
    }

    void FrameGrabberDirectory::nextFrame()
    {
        ++frame_index;
        setModified();
        step = true;
    }

    void FrameGrabberDirectory::prevFrame()
    {
        --frame_index;
        setModified();
        step = true;
    }

    bool FrameGrabberDirectory::processImpl()
    {
        if (synchronous && step == false)
        {
            return true;
        }

        step = false;
        if (fg)
        {
            if (frame_index < files_on_disk.size())
            {
                while (!fg->loadData(files_on_disk[frame_index]))
                {
                    if (!synchronous)
                        ++frame_index;
                    if (frame_index >= files_on_disk.size())
                    {
                        sig_eos();
                        return false;
                    }
                }
                if (!synchronous)
                {
                    ++frame_index;
                }
            }
            else if (frame_index >= files_on_disk.size())
            {
                sig_eos();
                return false;
            }
        }
        if (synchronous)
        {
            setModified(false);
        }
        return true;
    }

    void FrameGrabberDirectory::restart() { frame_index = 0; }

    bool FrameGrabberDirectory::loadData(std::string file_path)
    {
        auto path = boost::filesystem::path(file_path);
        if (boost::filesystem::exists(path) && boost::filesystem::is_directory(path))
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
            for (auto& itr : extension_map)
            {
                std::sort(itr.second.begin(), itr.second.end());
            }
            auto constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::IGrabber::getHash());
            std::vector<int> load_count(constructors.size(), 0);
            std::vector<int> priorities(constructors.size(), 0);
            for (int i = 0; i < constructors.size(); ++i)
            {
                if (auto info = dynamic_cast<const aq::nodes::GrabberInfo*>(constructors[i]->GetObjectInfo()))
                {
                    for (auto& file : extension_map)
                    {
                        int priority = info->canLoad(file.second[0]);
                        if (priority > 0)
                        {
                            ++load_count[i];
                            priorities[i] = priority;
                        }
                    }
                }
            }
            auto itr = std::max_element(priorities.begin(), priorities.end());
            if (itr != priorities.end() && (*itr) != 0)
            {
                long long idx = itr - priorities.begin();
                this->fg = constructors[idx]->Construct();
                this->fg->Init(true);
                const aq::nodes::GrabberInfo* info =
                    dynamic_cast<const aq::nodes::GrabberInfo*>(constructors[idx]->GetObjectInfo());
                for (auto& file : extension_map)
                {
                    if (info->canLoad(file.second[0]))
                    {
                        files_on_disk.insert(files_on_disk.end(), file.second.begin(), file.second.end());
                    }
                }
                this->addComponent(fg);
                if (files_on_disk.size() != 0)
                {
                    setModified();
                    sig_update();
                    step = true;
                    return true;
                }
            }
        }
        return false;
    }

} // namespace aqframegrabbers
using namespace aqframegrabbers;
MO_REGISTER_CLASS(FrameGrabberDirectory);
