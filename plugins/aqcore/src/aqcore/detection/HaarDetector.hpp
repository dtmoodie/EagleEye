#pragma once
#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA == 1
#include "../IDetector.hpp"
#include <Aquila/types/SyncedMemory.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect.hpp>

namespace aq
{
namespace nodes
{
class HaarDetector : public IImageDetector
{
  public:
    MO_DERIVE(HaarDetector, IImageDetector)
        PARAM(cv::Size, max_object_size, {200, 200})
        PARAM(cv::Size, min_object_size, {10, 10})
        PARAM(double, pyramid_scale_factor, 1.2)
        PARAM(int, min_neighbors, 3)
        PARAM(mo::ReadFile, model_file, {})
        PARAM(bool, use_gpu, true)
    MO_END

    template <class CTX>
    bool processImpl(CTX* ctx);

  protected:
    bool processImpl() override;
    cv::Ptr<cv::cuda::CascadeClassifier> m_gpu_detector;
    cv::Ptr<cv::CascadeClassifier> m_cpu_detector;
};
}
}
#endif