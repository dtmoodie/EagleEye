#pragma once
#include <aqcore_export.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedImage.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace aq
{
    namespace nodes
    {
        class HistogramEqualization : public Node
        {
          public:
            MO_DERIVE(HistogramEqualization, Node)
                INPUT(SyncedImage, input)

                PARAM(bool, per_channel, false)

                OUTPUT(SyncedImage, output)
            MO_END
          protected:
            bool processImpl();
        };
        class CLAHE : public Node
        {
          public:
            MO_DERIVE(CLAHE, Node)
                INPUT(SyncedImage, input)

                PARAM(double, clip_limit, 40)
                PARAM(int, grid_size, 8)

                OUTPUT(SyncedImage, output)
            MO_END
          protected:
            bool processImpl();
            cv::Ptr<cv::cuda::CLAHE> _clahe;
        };
    } // namespace nodes
} // namespace aq
