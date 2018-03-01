#pragma once

#include "../IDetector.hpp"
#include <Aquila/types/SyncedMemory.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect.hpp>
namespace aq
{
    namespace nodes
    {
        class HaarFaceDetector : public IImageDetector
        {
          public:
            MO_DERIVE(HaarFaceDetector, IImageDetector)
                INPUT(SyncedMemory, input, nullptr)
                PARAM(cv::Size, max_object_size, {200, 200})
                PARAM(cv::Size, min_object_size, {10, 10})
                PARAM(double, pyramid_scale_factor, 1.2)
                PARAM(int, min_neighbors, 3)
                PARAM(mo::ReadFile, model_file, {})
            MO_END;

          protected:
            virtual bool processImpl() override;
            cv::Ptr<cv::cuda::CascadeClassifier> m_gpu_detector;
            cv::Ptr<cv::CascadeClassifier> m_cpu_detector;
        };
    }
}
