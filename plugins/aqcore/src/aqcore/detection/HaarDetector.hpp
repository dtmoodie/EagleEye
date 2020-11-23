#pragma once
#include <MetaObject/core/metaobject_config.hpp>

#include "../IDetector.hpp"
#include "../OpenCVCudaNode.hpp"

#include <Aquila/types/SyncedMemory.hpp>

#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect.hpp>

namespace aqcore
{

    class HaarDetector : virtual public IImageDetector
    {
      public:
        MO_DERIVE(HaarDetector, IImageDetector)
            PARAM(cv::Size, max_object_size, {200, 200})
            PARAM(cv::Size, min_object_size, {10, 10})
            PARAM(double, pyramid_scale_factor, 1.2)
            PARAM(int, min_neighbors, 3)
            PARAM(mo::ReadFile, model_file, {})
            PARAM(bool, use_gpu, true)
        MO_END;

        bool processImpl(mo::IAsyncStream& stream) override;
        bool processImpl(mo::IDeviceStream& stream) override;

      protected:
        cv::Ptr<cv::cuda::CascadeClassifier> m_gpu_detector;
        cv::Ptr<cv::CascadeClassifier> m_cpu_detector;
    };
} // namespace aqcore
