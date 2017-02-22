#include "DetectionWriter.hpp"
#include "EagleLib/utilities/CudaCallbacks.hpp"
#include "EagleLib/Nodes/NodeInfo.hpp"

#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"

#include "cereal/archives/json.hpp"
#include <cereal/types/vector.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>

using namespace EagleLib;
using namespace EagleLib::Nodes;

bool DetectionWriter::ProcessImpl()
{
    if(output_directory_param.modified)
    {
        if(!boost::filesystem::exists(output_directory))
        {
            boost::filesystem::create_directories(output_directory);
        }else
        {
            // check if files exist, if they do, determine the current index and start appending
            boost::filesystem::path path(output_directory);
            boost::filesystem::directory_iterator end;
            for(boost::filesystem::directory_iterator itr(path); itr != end; ++itr)
            {
                std::string ext = itr->path().extension().string();

                if(ext == ".json")
                {
                    std::string stem = itr->path().stem().string();
                    if(stem.find(json_stem) == 0)
                    {
                        auto start = stem.find_first_of("0123456789");
                        auto end = stem.find_last_of("0123456789");
                        if(start != std::string::npos && end != std::string::npos)
                        {
                            int idx = boost::lexical_cast<int>(stem.substr(start, end));
                            frame_count = std::max(frame_count, idx + 1);
                        }
                    }
                }else if(ext == ".png")
                {
                    std::string stem = itr->path().stem().string();
                    if(stem.find(image_stem) == 0)
                    {
                        auto start = stem.find_first_of("0123456789");
                        auto end = stem.find_last_of("0123456789");
                        if(start != std::string::npos && end != std::string::npos)
                        {
                            int idx = boost::lexical_cast<int>(stem.substr(start, end));
                            frame_count = std::max(frame_count, idx + 1);
                        }
                    }
                }
            }
        }
        output_directory_param.modified = false;
    }

    bool found;
    if(object_class != -1)
    {
        found = false;
        for(const auto& detection : *detections)
        {
            if(detection.detections.size() && detection.detections[0].classNumber == object_class)
            {
                found = true;
                break;
            }
        }
    }else
    {
        if(detections->size() == 0)
            return false;
        found = true;
    }
    if(!found)
        return false;
    std::vector<DetectedObject> detections;// = *this->detections;
    for(const auto& detection : *this->detections)
    {
        if((detection.detections.size() &&
            detection.detections[0].classNumber == object_class) ||
            object_class == -1)
        {
            detections.push_back(detection);
        }
    }
    if(detections.size())
    {
        cv::Mat h_mat = image->GetMat(Stream());
        int fn = frame_count;

        cuda::enqueue_callback_async([h_mat, fn, this, detections]()
        {
            std::stringstream ss;
            ss << output_directory.string();
            ss << "/" << json_stem << "_" << std::setw(8) << std::setfill('0') << fn << ".json";
            std::ofstream ofs;
            ofs.open(ss.str());
            cereal::JSONOutputArchive ar(ofs);
            ss.str("");
            ss << output_directory.string() << "/" << image_stem << "_" << std::setw(8) << std::setfill('0') << fn << ".png";
            cv::imwrite(ss.str(), h_mat);
            ss.str("");
            ss << image_stem << "_" << std::setw(8) << std::setfill('0') << fn << ".png";
            ar(cereal::make_nvp("ImageFile", ss.str()));
            ar(cereal::make_nvp("Timestamp", image_param.GetTimestamp()));
            ar(cereal::make_nvp("detections",detections));

        }, _write_thread.GetId(), Stream());
        ++frame_count;
    }
    return true;
}

MO_REGISTER_CLASS(DetectionWriter)
