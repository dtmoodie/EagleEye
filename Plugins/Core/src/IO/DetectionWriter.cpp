#include "DetectionWriter.hpp"
#include "Aquila/ObjectDetectionSerialization.hpp"
#include "Aquila/utilities/CudaCallbacks.hpp"
#include "Aquila/Nodes/NodeInfo.hpp"
#include "MetaObject/Parameters/detail/TypedInputParameterPtrImpl.hpp"
#include "MetaObject/Parameters/detail/TypedParameterPtrImpl.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>

#include "cereal/archives/json.hpp"
#include <cereal/types/vector.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>

using namespace aq;
using namespace aq::Nodes;

int findNextIndex(const std::string& dir, const std::string& extension, const std::string& stem)
{
    boost::filesystem::path path(dir);
    boost::filesystem::directory_iterator end;
    int frame_count = 0;
    for(boost::filesystem::directory_iterator itr(path); itr != end; ++itr)
    {
        std::string ext = itr->path().extension().string();
        if(ext == extension)
        {
            std::string stem = itr->path().stem().string();
            if(stem.find(stem) == 0)
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
    return frame_count;
}

std::vector<DetectedObject> pruneDetections(const std::vector<DetectedObject>& input, int object_class)
{
    std::vector<DetectedObject> detections;
    bool found;
    if(object_class != -1)
    {
        found = false;
        for(const auto& detection : input)
        {
            if(detection.detections.size() && detection.detections[0].classNumber == object_class)
            {
                found = true;
                break;
            }
        }
    }else
    {
        if(input.size() == 0)
            return detections;
        found = true;
    }
    if(!found)
        return detections;

    for(const auto& detection : input)
    {
        if((detection.detections.size() &&
            detection.detections[0].classNumber == object_class) ||
            object_class == -1)
        {
            detections.push_back(detection);
        }
    }
    return detections;
}

bool DetectionWriter::ProcessImpl()
{
    if(output_directory_param._modified)
    {
        if(!boost::filesystem::exists(output_directory))
        {
            boost::filesystem::create_directories(output_directory);
        }else
        {
            // check if files exist, if they do, determine the current index and start appending
            int json_count = findNextIndex(output_directory.string(), ".json", json_stem);
            int img_count = findNextIndex(output_directory.string(), ".png", image_stem);
            frame_count = std::max(img_count, std::max(json_count, frame_count));
        }
        output_directory_param._modified = false;
    }
    auto detections = pruneDetections(*this->detections, object_class);

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
void DetectionWriterFolder::NodeInit(bool firstInit)
{
    if(firstInit)
    {
        _write_thread = boost::thread(
                    [this]()
        {
            std::pair<cv::Mat, std::string> data;
           while(!boost::this_thread::interruption_requested())
           {
               if(_write_queue.try_dequeue(data))
               {
                    cv::imwrite(data.second, data.first);
               }
           }
        });
    }
}
DetectionWriterFolder::~DetectionWriterFolder()
{
    if(_write_thread.joinable())
    {
        _write_thread.interrupt();
        _write_thread.join();
    }
}

bool DetectionWriterFolder::ProcessImpl()
{
    if(root_dir_param._modified)
    {
        for(int i = 0; i < labels->size(); ++i)
        {
            int frame_count = 0;
            if(!boost::filesystem::is_directory(root_dir.string() + "/" + (*labels)[i]))
            {
                boost::filesystem::create_directories(root_dir.string() + "/" + (*labels)[i]);
            }else
            {
                frame_count = findNextIndex(root_dir.string() + "/" + (*labels)[i], ".png", image_stem);
            }
            _frame_count = std::max(_frame_count, frame_count);
        }
        if(start_count != -1)
            _frame_count = start_count;
        root_dir_param._modified = false;
        _per_class_count.clear();
        _per_class_count.resize(labels->size(), 0);
        start_count = _frame_count;
    }
    auto detections = pruneDetections(*this->detections, object_class);
    if(image->GetSyncState() == image->DEVICE_UPDATED)
    {
        const cv::cuda::GpuMat img = image->GetGpuMat(Stream());
        cv::Rect img_rect(cv::Point(0,0), img.size());


        for(const aq::DetectedObject2d& detection : detections)
        {
            cv::Rect rect = img_rect & cv::Rect(detection.boundingBox.x - padding, detection.boundingBox.y - padding, detection.boundingBox.width + 2*padding, detection.boundingBox.height + 2*padding);
            std::string save_name;
            std::stringstream ss;
            cv::Mat save_img;
            img(rect).download(save_img, Stream());
            int idx = detection.detections[0].classNumber;
            ++_per_class_count[idx];
            {
                std::stringstream folderss;
                folderss << root_dir.string() << "/" << (*labels)[idx] << "/";
                folderss << std::setw(4) << std::setfill('0') << _per_class_count[idx] / max_subfolder_size;
                if(!boost::filesystem::is_directory(folderss.str()))
                {
                    boost::filesystem::create_directories(folderss.str());
                }
                ss << folderss.str() << "/";
            }

            ss << image_stem << std::setw(8) << std::setfill('0') << _frame_count++ << ".png";
            save_name = ss.str();
            cuda::enqueue_callback([this, save_img, save_name]()
            {
                this->_write_queue.enqueue(std::make_pair(save_img, save_name));
            }, Stream());
        }
    }else
    {
        cv::Mat img = image->GetMat(Stream());
        cv::Rect img_rect(cv::Point(0,0), img.size());
        for(const aq::DetectedObject2d& detection : detections)
        {
            cv::Rect rect = img_rect & cv::Rect(detection.boundingBox.x - padding, detection.boundingBox.y - padding, detection.boundingBox.width + 2*padding, detection.boundingBox.height + 2*padding);
            std::string save_name;
            std::stringstream ss;
            ss << root_dir.string() << "/" << (*labels)[detection.detections[0].classNumber] << "/" << image_stem << std::setw(8) << std::setfill('0') << _frame_count++ << ".png";
            save_name = ss.str();
            cuda::enqueue_callback([this, rect, img, save_name]()
            {
                cv::Mat save_img;
                img(rect).copyTo(save_img);
                this->_write_queue.enqueue(std::make_pair(save_img, save_name));
            }, Stream());
        }
    }
    return true;

}

MO_REGISTER_CLASS(DetectionWriterFolder)
