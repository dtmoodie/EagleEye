#include "DetectionWriter.hpp"

#include "Aquila/nodes/NodeInfo.hpp"
#include "Aquila/types/ObjectDetectionSerialization.hpp"
#include "Aquila/utilities/cuda/CudaCallbacks.hpp"
#include <Aquila/rcc/external_includes/cv_imgcodec.hpp>

#include "MetaObject/params/detail/TInputParamPtrImpl.hpp"
#include "MetaObject/params/detail/TParamPtrImpl.hpp"
#include "MetaObject/serialization/SerializationFactory.hpp"
#include <MetaObject/thread/boost_thread.hpp>
#include <ct/reflect/cereal.hpp>

#include "cereal/archives/json.hpp"
#include <cereal/types/vector.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <fstream>

using namespace aq;
using namespace aq::nodes;

int findNextIndex(const std::string& dir, const std::string& extension, const std::string& stem)
{
    boost::filesystem::path path(dir);
    boost::filesystem::directory_iterator end;
    int frame_count = 0;
    for (boost::filesystem::directory_iterator itr(path); itr != end; ++itr)
    {
        std::string ext = itr->path().extension().string();
        if (ext == extension)
        {
            std::string stem = itr->path().stem().string();
            if (stem.find(stem) == 0)
            {
                auto start = stem.find_last_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_");
                auto end = stem.find_last_of("0123456789");
                if (stem.size() && end != std::string::npos)
                {
                    int idx = boost::lexical_cast<int>(stem.substr(start == std::string::npos ? 0 : start + 1, end));
                    frame_count = std::max(frame_count, idx + 1);
                }
            }
        }
    }
    return frame_count;
}

DetectedObjectSet pruneDetections(const DetectedObjectSet& input, int object_class)
{
    DetectedObjectSet detections;
    bool found;
    if (object_class != -1)
    {
        found = false;
        for (const auto& detection : input)
        {
            if (detection.classifications[0].cat->index == object_class)
            {
                found = true;
                break;
            }
        }
    }
    else
    {
        if (input.size() == 0)
            return detections;
        found = true;
    }
    if (!found)
        return detections;

    for (const auto& detection : input)
    {
        if ((detection.classifications[0].cat->index == object_class) || object_class == -1)
        {
            detections.emplace_back(detection);
        }
    }
    return detections;
}

IDetectionWriter::~IDetectionWriter()
{
    if (_write_thread && !IsRuntimeDelete())
    {
        _write_thread->interrupt();
        _write_thread->join();
    }
}

void IDetectionWriter::nodeInit(bool firstInit)
{
    if (firstInit)
    {
        _write_thread.reset(new boost::thread(&IDetectionWriter::writeThread, this));
    }
}

/*bool IDetectionWriter::processImpl()
{
    if (output_directory_param.modified())
    {
        if (!boost::filesystem::exists(output_directory))
        {
            boost::filesystem::create_directories(output_directory);
        }
        else
        {
            // check if files exist, if they do, determine the current index and start appending
            int json_count = findNextIndex(output_directory.string(), ".json", annotation_stem);
            int img_count = findNextIndex(output_directory.string(), "." + extension.getEnum(), image_stem);
            frame_count = std::max<size_t>(img_count, std::max<size_t>(json_count, frame_count));
        }
        output_directory_param.modified(false);
    }
    auto detections = pruneDetections(*this->detections, object_class);

    if (detections.size() || skip_empty == false)
    {
        cv::Mat h_mat = image->getMat(stream());
        //cuda::enqueue_callback(
          //  [h_mat, this, detections]() { this->_write_queue->enqueue(std::make_pair(h_mat, detections)); },
stream());
    }
    return true;
}*/

void IDetectionWriter::writeThread()
{
    std::function<void(void)> work;
    while (!boost::this_thread::interruption_requested())
    {
        if (_write_queue.try_dequeue(work))
        {
            work();
        }
        else
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(5));
        }
    }
}

bool DetectionWriter::processImpl()
{
    if (output_directory_param.modified())
    {
        if (!boost::filesystem::exists(output_directory))
        {
            boost::filesystem::create_directories(output_directory);
        }
        else
        {
            // check if files exist, if they do, determine the current index and start appending
            int json_count = findNextIndex(output_directory.string(), ".json", annotation_stem);
            int img_count = findNextIndex(output_directory.string(), "." + extension.getEnum(), image_stem);
            frame_count = std::max<size_t>(img_count, std::max<size_t>(json_count, frame_count));
        }
        output_directory_param.modified(false);
    }
    if (detections->size())
    {
        auto dets = pruneDetections(*detections, object_class);
        cv::Mat h_mat = image->getMat(stream());
        auto ts = image_param.getTimestamp();
        auto fn = image_param.getFrameNumber();
        size_t count = frame_count;
        cuda::enqueue_callback(
            [count, ts, fn, h_mat, dets, this]() {
                this->_write_queue.enqueue([count, ts, fn, h_mat, dets, this]() {
                    std::stringstream ss;
                    ss << output_directory.string();
                    ss << "/" << annotation_stem << std::setw(8) << std::setfill('0') << count << ".json";
                    std::ofstream ofs;
                    ofs.open(ss.str());
                    cereal::JSONOutputArchive ar(ofs);
                    ss.str(std::string()); // = std::stringstream();
                    ss << output_directory.string() << "/" << image_stem << std::setw(8) << std::setfill('0') << count
                       << "." << extension.getEnum();
                    cv::imwrite(ss.str(), h_mat);
                    ss.str(std::string()); // = std::stringstream();
                    ss << image_stem << std::setw(8) << std::setfill('0') << frame_count << "." << extension.getEnum();
                    ar(cereal::make_nvp("ImageFile", ss.str()));
                    if (ts)
                        ar(cereal::make_nvp("timestamp", *ts));
                    ar(cereal::make_nvp("framenumber", fn));
                    ar(cereal::make_nvp("detections", dets));
                });
            },
            stream());
        ++frame_count;
    }
    return true;
}

/*WriteData_t data;
mo::setThisThreadName("DetectionWriter");
while (!boost::this_thread::interruption_requested())
{
    if (this->_write_queue->try_dequeue(data))
    {

        if (pad)

        else
            ss << "/" << annotation_stem << frame_count << ".json";
        std::ofstream ofs;
        ofs.open(ss.str());
        cereal::JSONOutputArchive ar(ofs);
        ss.str("");
        if (pad)
            ss << output_directory.string() << "/" << image_stem << std::setw(8) << std::setfill('0') << frame_count
               << "." << extension.getEnum();
        else
            ss << output_directory.string() << "/" << image_stem << frame_count << "." << extension.getEnum();
        cv::imwrite(ss.str(), data.first);
        ss.str("");
        if (pad)
            ss << image_stem << std::setw(8) << std::setfill('0') << frame_count << "." << extension.getEnum();
        else
            ss << image_stem << frame_count << "." << extension.getEnum();
        ar(cereal::make_nvp("ImageFile", ss.str()));
        ar(cereal::make_nvp("detections", data.second));
        ++frame_count;
    }
}*/

MO_REGISTER_CLASS(DetectionWriter)

void DetectionWriterFolder::nodeInit(bool firstInit)
{
    if (firstInit)
    {
        _write_thread = boost::thread([this]() {
            std::pair<cv::Mat, std::string> data;
            while (!boost::this_thread::interruption_requested())
            {
                if (_write_queue.try_dequeue(data))
                {
                    cv::imwrite(data.second, data.first);
                }
            }
        });
    }
}

DetectionWriterFolder::~DetectionWriterFolder()
{
    if (_write_thread.joinable())
    {
        _write_thread.interrupt();
        _write_thread.join();
    }
    _summary_ar.reset();
    _summary_ofs.reset();
}

struct FrameDetections
{
    FrameDetections(const aq::DetectedObject& det) : detections(det) {}

    std::string source_path;
    mo::Time_t timestamp;
    size_t frame_number;
    const aq::DetectedObject& detections;
    std::vector<std::string> written_detections;
    template <class AR>
    void serialize(AR& ar)
    {
        ar(CEREAL_NVP(source_path));
        ar(CEREAL_NVP(timestamp));
        ar(CEREAL_NVP(frame_number));
        ar(CEREAL_NVP(detections));
    }
};

struct WritePair
{
    WritePair(const DetectedObject& det, const std::string& name) : detection(det), patch_name(name) {}
    DetectedObject detection;
    std::string patch_name;
    template <class AR>
    void serialize(AR& ar)
    {
        ar(CEREAL_NVP(detection), CEREAL_NVP(patch_name));
    }
};

bool DetectionWriterFolder::processImpl()
{
    if (!_summary_ofs)
    {
        if (!boost::filesystem::is_directory(root_dir))
        {
            boost::filesystem::create_directories(root_dir);
        }
        _summary_ofs.reset(new std::ofstream());
        _summary_ofs->open(root_dir.string() + "/summary.json");
        _summary_ar.reset(new cereal::JSONOutputArchive(*_summary_ofs));
        (*_summary_ar)(CEREAL_NVP(dataset_name));
    }
    if (root_dir_param.modified())
    {
        auto cats = detections->getCatSet();
        for (int i = 0; i < cats->size(); ++i)
        {
            int frame_count = 0;
            if (!boost::filesystem::is_directory(root_dir.string() + "/" + (*cats)[i].name))
            {
                boost::filesystem::create_directories(root_dir.string() + "/" + (*cats)[i].name);
            }
            else
            {
                frame_count =
                    findNextIndex(root_dir.string() + "/" + (*cats)[i].name, "." + extension.getEnum(), image_stem);
            }
            _frame_count = std::max(_frame_count, frame_count);
        }
        if (start_count != -1)
            _frame_count = start_count;
        root_dir_param.modified(false);
        _per_class_count.clear();
        _per_class_count.resize(cats->size(), 0);
        start_count = _frame_count;
    }

    DetectedObjectSet detections;
    if (this->detections)
    {
        detections = pruneDetections(*this->detections, object_class);
    }

    std::vector<WritePair> written_detections;
    if (image->getSyncState() == image->DEVICE_UPDATED)
    {
        const cv::cuda::GpuMat img = image->getGpuMat(stream());
        cv::Rect img_rect(cv::Point(0, 0), img.size());
        for (const auto& detection : detections)
        {
            cv::Rect rect = img_rect & cv::Rect(detection.bounding_box.x - padding,
                                                detection.bounding_box.y - padding,
                                                detection.bounding_box.width + 2 * padding,
                                                detection.bounding_box.height + 2 * padding);
            std::string save_name;
            std::stringstream ss;
            cv::Mat save_img;
            img(rect).download(save_img, stream());
            const std::string& name = detection.classifications[0].cat->name;
            unsigned int idx = detection.classifications[0].cat->index;
            ++_per_class_count[idx];
            {
                std::stringstream folderss;
                folderss << root_dir.string() << "/" << name << "/";
                folderss << std::setw(4) << std::setfill('0') << _per_class_count[idx] / max_subfolder_size;
                if (!boost::filesystem::is_directory(folderss.str()))
                {
                    boost::filesystem::create_directories(folderss.str());
                }
                ss << folderss.str() << "/";
            }
            ++_frame_count;
            ss << image_stem << std::setw(8) << std::setfill('0') << _frame_count << "." + extension.getEnum();
            save_name = ss.str();
            cuda::enqueue_callback(
                [this, save_img, save_name]() { this->_write_queue.enqueue(std::make_pair(save_img, save_name)); },
                stream());
            ss.str(std::string());
            ss << name << "/" << std::setw(4) << std::setfill('0')
               << _per_class_count[idx] / max_subfolder_size;
            ss << image_stem << std::setw(8) << std::setfill('0') << _frame_count << "." + extension.getEnum();
            save_name = ss.str();
            written_detections.emplace_back(detection, save_name);
        }
    }
    else
    {
        cv::Mat img = image->getMat(stream());
        cv::Rect img_rect(cv::Point(0, 0), img.size());
        for (const auto& detection : detections)
        {
            cv::Rect rect = img_rect & cv::Rect(detection.bounding_box.x - padding,
                                                detection.bounding_box.y - padding,
                                                detection.bounding_box.width + 2 * padding,
                                                detection.bounding_box.height + 2 * padding);
            std::string save_name;
            std::stringstream ss;
            const std::string& name = detection.classifications[0].cat->name;
            unsigned int idx = detection.classifications[0].cat->index;
            ++_per_class_count[idx];
            {
                std::stringstream folderss;
                folderss << root_dir.string() << "/" << name << "/";
                folderss << std::setw(4) << std::setfill('0') << _per_class_count[idx] / max_subfolder_size;
                if (!boost::filesystem::is_directory(folderss.str()))
                {
                    boost::filesystem::create_directories(folderss.str());
                }
                ss << folderss.str() << "/";
            }

            ss << image_stem << std::setw(8) << std::setfill('0') << _frame_count++ << "." + extension.getEnum();
            save_name = ss.str();
            cuda::enqueue_callback(
                [this, rect, img, save_name]() {
                    cv::Mat save_img; //(cv::Mat::getStdAllocator());
                    save_img.allocator = cv::Mat::getStdAllocator();
                    img(rect).copyTo(save_img);
                    this->_write_queue.enqueue(std::make_pair(save_img, save_name));
                },
                stream());
        }
    }
    if (written_detections.size())
    {
        (*_summary_ar)(written_detections);
    }
    return true;
}

MO_REGISTER_CLASS(DetectionWriterFolder)
