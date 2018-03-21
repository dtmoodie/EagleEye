#include "gstreamer.h"
#include "precompiled.hpp"
#include <Aquila/framegrabbers/GrabberInfo.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace aq;
using namespace aq::nodes;

int GrabberGstreamer::canLoad(const std::string& path)
{
    // oooor a gstreamer pipeline....
    std::string appsink = "appsink";
    if (path.find(appsink) != std::string::npos)
        return 9;
    if (boost::filesystem::exists(path))
        return 2;
    MO_LOG(trace) << "Document is not a regular file";
    return 0;
}
void GrabberGstreamer::listPaths(std::vector<std::string>& paths)
{
    if (boost::filesystem::exists("file_history.json"))
    {
        boost::property_tree::ptree file_history;
        boost::property_tree::json_parser::read_json("file_history.json", file_history);
        auto files = file_history.get_child_optional("files");
        if (files)
        {
            for (auto itr = files->begin(); itr != files->end(); ++itr)
            {
                auto path = itr->second.get<std::string>("path", "");
                if (path.size())
                    paths.push_back(path);
            }
        }
    }
}

bool GrabberGstreamer::loadData(const std::string& file_path_)
{
    std::string file_path = file_path_;

    h_cam.reset(new cv::VideoCapture(file_path_, cv::CAP_GSTREAMER));
    if (h_cam->isOpened())
    {
        loaded_document = file_path;
        cv::Mat test;

        if (!h_cam->read(test))
        {
            return false;
        }
        h_cam->set(cv::CAP_PROP_POS_FRAMES, 0);
        boost::property_tree::ptree file_history;
        if (boost::filesystem::exists("file_history.json"))
        {
            boost::property_tree::json_parser::read_json("file_history.json", file_history);
        }
        boost::property_tree::ptree child;
        child.put("path", file_path);
        if (!file_history.get_child_optional("files"))
            file_history.add_child("files", boost::property_tree::ptree());
        for (auto& paths : file_history.get_child("files"))
        {
            (void)paths;
            if (child.get_child("path").get_value<std::string>() == file_path)
            {
                return true;
            }
        }
        file_history.get_child("files").push_back(std::make_pair("", child));
        boost::property_tree::json_parser::write_json("file_history.json", file_history);
        return true;
    }
    return false;
}

MO_REGISTER_CLASS(GrabberGstreamer);
