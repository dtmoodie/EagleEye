#pragma once

#include <EagleLib/Nodes/Node.h>
#include <opencv2/cudaarithm.hpp>
#include "../CoreExport.hpp"
#include <EagleLib/rcc/external_includes/cv_cudafilters.hpp>
namespace EagleLib
{
    Core_EXPORT void applyColormap(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const cv::cuda::GpuMat& colormap, cv::cuda::Stream& stream);


    namespace Nodes
    {
        class LabelDisplay: public Node
        {
        public:
            MO_DERIVE(LabelDisplay, Node)
                INPUT(SyncedMemory, label, nullptr)
                OPTIONAL_INPUT(SyncedMemory, original_image, nullptr)
                OUTPUT(SyncedMemory, colorized, SyncedMemory())
                PARAM(int, num_classes, 13)
                PARAM(float, label_weight, 0.7)
                PARAM(int, ignore_class, -1)
                PARAM(bool, display_legend, true)
                PARAM(mo::ReadFile, label_file, {})
                PARAM(int, dilate, 0)
            MO_END
        protected:
            bool ProcessImpl();
            cv::cuda::GpuMat d_lut;
            cv::cuda::GpuMat d_legend;
            cv::Mat h_lut;
            std::vector<std::string> labels;
            int legend_width;
            cv::Ptr<cv::cuda::Filter> _dilate_filter;
        };
    }
}
