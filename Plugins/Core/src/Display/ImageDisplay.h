#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/Stamped.hpp>
#include "../CoreExport.hpp"

namespace cv
{
namespace cuda
{
    void Core_EXPORT drawHistogram(cv::InputArray histogram,
                       cv::OutputArray draw,
                       cv::InputArray bins = cv::noArray(),
                       cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    void Core_EXPORT drawPlot(cv::InputArray plot, // 1D array
                       cv::OutputArray draw,
                       cv::cuda::Stream& stream = cv::cuda::Stream::Null());
}
}
namespace aq
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
        MO_DERIVE(OGLImageDisplay, Node)
            INPUT(SyncedMemory, image, nullptr)
        MO_END;
        bool ProcessImpl();
    protected:
        boost::optional<mo::time_t> _prev_time;
    };


    class KeyPointDisplay: public Node
    {
    public:
        MO_DERIVE(KeyPointDisplay, Node)
            INPUT(TS<SyncedMemory>, image, nullptr)
            INPUT(TS<SyncedMemory>, synced_points,  nullptr)
            INPUT(cv::cuda::GpuMat, gpu_points, nullptr)
            INPUT(cv::Mat, cpu_points, nullptr)
        MO_END;
    protected:
        bool ProcessImpl();
    };
    class FlowVectorDisplay: public Node
    {
    public:
        MO_DERIVE(FlowVectorDisplay, Node)
            INPUT(TS<SyncedMemory>, image, nullptr)
        MO_END;
    protected:
        bool ProcessImpl();
    };

    class HistogramDisplay: public Node
    {
    public:
        MO_DERIVE(HistogramDisplay, Node)
            INPUT(SyncedMemory, histogram, nullptr)
            OPTIONAL_INPUT(SyncedMemory, bins, nullptr)
        MO_END;
    protected:
        bool ProcessImpl();
        cv::cuda::GpuMat draw;
    };
    class HistogramOverlay: public Node
    {
    public:
        MO_DERIVE(HistogramOverlay, Node)
            INPUT(SyncedMemory, histogram, nullptr)
            INPUT(SyncedMemory, image, nullptr)
            OPTIONAL_INPUT(SyncedMemory, bins, nullptr)
            OUTPUT(SyncedMemory, output, {})
        MO_END;
    protected:
        bool ProcessImpl();
        cv::cuda::GpuMat draw;
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
} // namespace aq
