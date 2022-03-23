#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/rcc/external_includes/cv_cudafilters.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <aqcore/aqcore_export.hpp>
#include <opencv2/cudaarithm.hpp>
namespace aq
{
    aqcore_EXPORT void applyColormap(const cv::cuda::GpuMat& input,
                                     cv::cuda::GpuMat& output,
                                     const cv::cuda::GpuMat& colormap,
                                     cv::cuda::Stream& stream);

    namespace nodes
    {
        class LabelDisplay : public Node
        {
          public:
            MO_DERIVE(LabelDisplay, Node)
                INPUT(SyncedMemory, label, nullptr)
                OPTIONAL_INPUT(SyncedMemory, original_image, nullptr)
                INPUT(std::vector<std::string>, labels, nullptr)
                OUTPUT(SyncedMemory, colorized, SyncedMemory())
                PARAM(float, label_weight, 0.7f)
                PARAM(int, ignore_class, -1)
                PARAM(bool, display_legend, true)
                PARAM(int, dilate, 0)
            MO_END
          protected:
            bool processImpl();
            cv::cuda::GpuMat d_lut;
            cv::cuda::GpuMat d_legend;
            cv::Mat h_lut;
            // std::vector<std::string> labels;
            int legend_width;
            cv::Ptr<cv::cuda::Filter> _dilate_filter;
        };
    } // namespace nodes
} // namespace aq
