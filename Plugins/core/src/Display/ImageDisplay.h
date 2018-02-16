#include "aqcore_export.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/Stamped.hpp>
#include <Aquila/types/SyncedMemory.hpp>

namespace cv
{
    namespace cuda
    {
        void aqcore_EXPORT drawHistogram(cv::InputArray histogram,
                                         cv::OutputArray draw,
                                         cv::InputArray bins = cv::noArray(),
                                         cv::cuda::Stream& stream = cv::cuda::Stream::Null());

        void aqcore_EXPORT drawPlot(cv::InputArray plot, // 1D array
                                    cv::OutputArray draw,
                                    cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    }
}
namespace aq
{
    namespace nodes
    {
        class QtImageDisplay : public Node
        {
          public:
            MO_DERIVE(QtImageDisplay, Node)
                OPTIONAL_INPUT(SyncedMemory, image, nullptr)
                OPTIONAL_INPUT(cv::Mat, cpu_mat, nullptr)
                PARAM(bool, overlay_timestamp, true)
            MO_END
          protected:
            bool processImpl();
        };

        class OGLImageDisplay : public Node
        {
          public:
            MO_DERIVE(OGLImageDisplay, Node)
                INPUT(SyncedMemory, image, nullptr)
            MO_END;
            bool processImpl();

          protected:
            boost::optional<mo::Time_t> _prev_time;
            bool m_use_opengl = true;
        };

        class KeyPointDisplay : public Node
        {
          public:
            MO_DERIVE(KeyPointDisplay, Node)
                INPUT(TS<SyncedMemory>, image, nullptr)
                INPUT(TS<SyncedMemory>, synced_points, nullptr)
                INPUT(cv::cuda::GpuMat, gpu_points, nullptr)
                INPUT(cv::Mat, cpu_points, nullptr)
            MO_END;

          protected:
            bool processImpl();
        };

        class FlowVectorDisplay : public Node
        {
          public:
            MO_DERIVE(FlowVectorDisplay, Node)
                INPUT(TS<SyncedMemory>, image, nullptr)
            MO_END;

          protected:
            bool processImpl();
        };

        class HistogramDisplay : public Node
        {
          public:
            MO_DERIVE(HistogramDisplay, Node)
                INPUT(SyncedMemory, histogram, nullptr)
                OPTIONAL_INPUT(SyncedMemory, bins, nullptr)
            MO_END;

          protected:
            bool processImpl();
            cv::cuda::GpuMat draw;
        };

        class HistogramOverlay : public Node
        {
          public:
            MO_DERIVE(HistogramOverlay, Node)
                INPUT(SyncedMemory, histogram, nullptr)
                INPUT(SyncedMemory, image, nullptr)
                OPTIONAL_INPUT(SyncedMemory, bins, nullptr)
                OUTPUT(SyncedMemory, output, {})
            MO_END;

          protected:
            bool processImpl();
            cv::cuda::GpuMat draw;
        };

        class DetectionDisplay : public Node
        {
          public:
            MO_DERIVE(DetectionDisplay, Node)
                ;
                INPUT(TS<SyncedMemory>, image, nullptr);
            MO_END;

          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
