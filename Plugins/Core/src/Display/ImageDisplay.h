#include "src/precompiled.hpp"
#include "EagleLib/Nodes/Sink.h"
#include "src/precompiled.hpp"

namespace EagleLib
{
namespace Nodes
{
    class QtImageDisplay: public Node
    {
    public:
        MO_DERIVE(QtImageDisplay, Node)
            OPTIONAL_INPUT(SyncedMemory, image, nullptr)
            OPTIONAL_INPUT(cv::Mat, cpu_mat, nullptr)
            PARAM(bool, overlay_timestamp, true)
        MO_END
    protected:
        bool ProcessImpl();
    };
    class OGLImageDisplay: public Node
    {
    public:
        MO_DERIVE(OGLImageDisplay, Node);
            INPUT(TS<SyncedMemory>, image, nullptr);
        MO_END;
        bool ProcessImpl();
    };
    class KeyPointDisplay: public Node
    {
    public:
        MO_DERIVE(KeyPointDisplay, Node);
            INPUT(TS<SyncedMemory>, image, nullptr);
            INPUT(TS<SyncedMemory>, synced_points,  nullptr);
            INPUT(cv::cuda::GpuMat, gpu_points, nullptr);
            INPUT(cv::Mat, cpu_points, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class FlowVectorDisplay: public Node
    {
    public:
        MO_DERIVE(FlowVectorDisplay, Node);
            INPUT(TS<SyncedMemory>, image, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };

    class HistogramDisplay: public Node
    {
    public:
        MO_DERIVE(HistogramDisplay, Node);
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class DetectionDisplay: public Node
    {
    public:
        MO_DERIVE(DetectionDisplay, Node);
        INPUT(TS<SyncedMemory>, image, nullptr);
        MO_END;
    protected:
        bool ProcessImpl();
    };
} // namespace Nodes
} // namespace EagleLib
