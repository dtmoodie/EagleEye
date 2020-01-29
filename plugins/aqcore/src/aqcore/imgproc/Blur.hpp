#pragma once
#include "Aquila/rcc/external_includes/cv_cudafilters.hpp"
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <aqcore/aqcore_export.hpp>
namespace cv
{
namespace cuda
{
class Filter;
}
}

namespace aq
{
namespace nodes
{
class MedianBlur : public Node
{
  public:
    MO_DERIVE(MedianBlur, Node)
        INPUT(SyncedMemory, input, nullptr)
        PARAM(int, window_size, 5)
        PARAM(int, partition, 128)
        OUTPUT(SyncedMemory, output, {})
    MO_END
  protected:
    virtual bool processImpl() override;

    cv::Ptr<cv::cuda::Filter> _median_filter;
};

class aqcore_EXPORT GaussianBlur : public Node
{
  public:
    MO_DERIVE(GaussianBlur, Node)
        INPUT(SyncedMemory, input, nullptr)
        PARAM(int, kerenl_size, 5)
        PARAM(double, sigma, 1.0)
        OUTPUT(SyncedMemory, output, {})
    MO_END
    virtual bool processImpl() override;

    template <class CType>
    bool processImpl(CType* ctx);

  protected:
    cv::Ptr<cv::cuda::Filter> _blur_filter;
};
}
}
