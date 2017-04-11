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
class DetectionWriter: public Node
{
public:
    MO_DERIVE(DetectionWriter, Node)
        PARAM(mo::WriteDirectory, output_directory, {})
        PARAM(std::string, json_stem, "detection")
        PARAM(std::string, image_stem, "image")
        PARAM(int, object_class, -1)
        INPUT(SyncedMemory, image, nullptr)
        INPUT(std::vector<DetectedObject>, detections, nullptr)
        PROPERTY(mo::ThreadHandle, _write_thread, mo::ThreadPool::Instance()->RequestThread())
    MO_END
protected:
    bool ProcessImpl();
    int frame_count = 0;
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
        INPUT(SyncedMemory, image, nullptr)
        INPUT(std::vector<std::string>, labels, nullptr)
        INPUT(std::vector<DetectedObject>, detections, nullptr)
        PARAM(int, start_count, -1)
    MO_END;

protected:
    void NodeInit(bool firstInit);
    bool ProcessImpl();
    //std::vector<int> _frame_counts;
    int _frame_count;
    moodycamel::ConcurrentQueue<std::pair<cv::Mat, std::string>> _write_queue;
    boost::thread _write_thread;
    std::vector<int> _per_class_count;

};

}
}
