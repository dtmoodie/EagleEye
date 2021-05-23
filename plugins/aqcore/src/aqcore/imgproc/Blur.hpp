#pragma once
#include <aqcore_export.hpp>

#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

#include <Aquila/rcc/external_includes/cv_cudafilters.hpp>

namespace cv
{
    namespace cuda
    {
        class Filter;
    }
} // namespace cv

namespace mo
{
    namespace cuda
    {
        struct AsyncStream;
    }
} // namespace mo

namespace aq
{
    namespace nodes
    {
        class MedianBlur : public Node
        {
          public:
            MO_DERIVE(MedianBlur, Node)
                INPUT(SyncedImage, input)
                PARAM(int, window_size, 5)
                PARAM(int, partition, 128)
                OUTPUT(SyncedMemory, output)
            MO_END;

          protected:
            bool processImpl() override;

            cv::Ptr<cv::cuda::Filter> _median_filter;
        };

        class aqcore_EXPORT GaussianBlur : public Node
        {
          public:
            MO_DERIVE(GaussianBlur, Node)
                INPUT(SyncedImage, input)

                PARAM(int, kerenl_size, 5)
                PARAM(double, sigma, 1.0)

                OUTPUT(SyncedImage, output, {})
            MO_END;

            bool processImpl() override;

            bool processImpl(mo::IAsyncStream&);
            bool processImpl(mo::cuda::AsyncStream&);

          protected:
            cv::Ptr<cv::cuda::Filter> _blur_filter;
        };
    } // namespace nodes
} // namespace aq
