#pragma once

#include "Aquila/Nodes/Node.h"
#include "Aquila/ObjectDetection.hpp"
#include "MetaObject/Thread/ThreadHandle.hpp"
#include "MetaObject/Thread/ThreadPool.hpp"
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
}
}
