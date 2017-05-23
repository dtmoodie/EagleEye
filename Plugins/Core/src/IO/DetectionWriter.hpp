#pragma once

#include "Aquila/Nodes/Node.h"
#include "Aquila/ObjectDetection.hpp"
#include "MetaObject/Thread/ThreadHandle.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"
#include "MetaObject/Detail/ConcurrentQueue.hpp"
namespace aq
{
namespace Nodes
{
enum Extension{
    jpg,
    png,
    tiff,
    bmp
};

class IDetectionWriter: public Node{
public:
    typedef std::pair<cv::Mat, std::vector<aq::DetectedObject2d>> WriteData_t;
    typedef moodycamel::ConcurrentQueue<WriteData_t> WriteQueue_t;
    ~IDetectionWriter();
    MO_DERIVE(IDetectionWriter, Node)
        PARAM(mo::WriteDirectory, output_directory, {})
        PARAM(std::string, annotation_stem, "detection")
        PARAM(std::string, image_stem, "image")
        PARAM(int, object_class, -1)
        PARAM(bool, skip_empty, true)
        PARAM(bool, pad, true)
        ENUM_PARAM(extension, jpg, png, tiff, bmp)
        INPUT(SyncedMemory, image, nullptr)
        INPUT(std::vector<DetectedObject>, detections, nullptr)
        PROPERTY(std::shared_ptr<boost::thread>, _write_thread, {})
        PROPERTY(std::shared_ptr<WriteQueue_t>, _write_queue, {})
    MO_END
protected:
    bool ProcessImpl();
    void NodeInit(bool firstInit);
    virtual void writeThread() = 0;
    size_t frame_count = 0;
};


class DetectionWriter: public IDetectionWriter
{
public:
    MO_DERIVE(DetectionWriter, IDetectionWriter)
    MO_END
protected:
    virtual void writeThread();
};

class DetectionWriterFolder: public Node
{
public:
    ~DetectionWriterFolder();
    MO_DERIVE(DetectionWriterFolder, Node)
        PARAM(mo::WriteDirectory, root_dir, {})
        PARAM(int, padding, 0)
        PARAM(int, object_class, -1)
        PARAM(std::string, image_stem, "image")
        PARAM(int, max_subfolder_size, 1000)
        PARAM(std::string, dataset_name, "")
        ENUM_PARAM(extension, jpg, png, tiff, bmp)
        INPUT(SyncedMemory, image, nullptr)
        INPUT(std::vector<std::string>, labels, nullptr)
        INPUT(std::vector<DetectedObject>, detections, nullptr)
        PARAM(int, start_count, -1)
    MO_END;

protected:
    void NodeInit(bool firstInit);
    bool ProcessImpl();
    int _frame_count;
    moodycamel::ConcurrentQueue<std::pair<cv::Mat, std::string>> _write_queue;
    boost::thread _write_thread;
    std::vector<int> _per_class_count;
    std::shared_ptr<std::ofstream> _summary_ofs;
    std::shared_ptr<cereal::JSONOutputArchive> _summary_ar;
};

}
}
