#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include "../OpenCVCudaNode.hpp"
#include <Aquila/nodes/Node.hpp>

namespace aqcore
{

    class MaskOverlay : public OpenCVCudaNode
    {
      public:
        static const cv::Scalar default_color;

        MO_DERIVE(MaskOverlay, OpenCVCudaNode)
            INPUT(aq::SyncedImage, image)
            INPUT(aq::SyncedImage, mask)

            PARAM(cv::Scalar, color, default_color)

            OUTPUT(aq::SyncedImage, output)
        MO_END;

      protected:
        virtual bool processImpl() override;
    };

} // namespace aqcore
