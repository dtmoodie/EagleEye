#include "aqcore/aqcore_export.hpp"

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

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
    } // namespace cuda
} // namespace cv
namespace aq
{
    namespace nodes
    {
        class aqcore_EXPORT QtImageDisplay : public Node
        {
          public:
            MO_DERIVE(QtImageDisplay, Node)
                OPTIONAL_INPUT(SyncedImage, image)
                OPTIONAL_INPUT(cv::Mat, cpu_mat)
                PARAM(bool, overlay_timestamp, true)
            MO_END;

          protected:
            bool processImpl();
        };

        class OGLImageDisplay : public Node
        {
          public:
            MO_DERIVE(OGLImageDisplay, Node)
                INPUT(SyncedImage, image)
            MO_END;
            bool processImpl();

          protected:
            mo::OptionalTime m_prev_time;
            bool m_use_opengl = true;
        };

        class KeyPointDisplay : public Node
        {
          public:
            MO_DERIVE(KeyPointDisplay, Node)
                INPUT(SyncedImage, image)
                INPUT(SyncedImage, synced_points)
                INPUT(cv::cuda::GpuMat, gpu_points)
                INPUT(cv::Mat, cpu_points)
            MO_END;

          protected:
            bool processImpl();
        };

        class FlowVectorDisplay : public Node
        {
          public:
            MO_DERIVE(FlowVectorDisplay, Node)
                INPUT(SyncedImage, image)
            MO_END;

          protected:
            bool processImpl();
        };

        class HistogramDisplay : public Node
        {
          public:
            MO_DERIVE(HistogramDisplay, Node)
                INPUT(SyncedImage, histogram)
                OPTIONAL_INPUT(SyncedImage, bins)
            MO_END;

          protected:
            bool processImpl();
            cv::cuda::GpuMat draw;
        };

        class HistogramOverlay : public Node
        {
          public:
            MO_DERIVE(HistogramOverlay, Node)
                INPUT(SyncedImage, histogram)
                INPUT(SyncedImage, image)
                OPTIONAL_INPUT(SyncedImage, bins)
                OUTPUT(SyncedImage, output)
            MO_END;

          protected:
            bool processImpl();
            cv::cuda::GpuMat draw;
        };

        class DetectionDisplay : public Node
        {
          public:
            MO_DERIVE(DetectionDisplay, Node)
                INPUT(SyncedImage, image)
            MO_END;

          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
