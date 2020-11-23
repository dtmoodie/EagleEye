#include <MetaObject/core/metaobject_config.hpp>

#include "HaarDetector.hpp"

#include <boost/filesystem.hpp>

namespace aqcore
{

    bool HaarDetector::processImpl(mo::IDeviceStream& stream)
    {
        if (!m_gpu_detector || model_file_param.getModified())
        {
            m_gpu_detector = cv::cuda::CascadeClassifier::create(model_file.string());
            model_file_param.setModified(false);
        }
        if (max_object_size_param.getModified())
        {
            m_gpu_detector->setMaxObjectSize(max_object_size);
            max_object_size_param.setModified(false);
        }
        if (min_object_size_param.getModified())
        {
            m_gpu_detector->setMinObjectSize(min_object_size);
            min_object_size_param.setModified(false);
        }
        if (pyramid_scale_factor_param.getModified())
        {
            m_gpu_detector->setScaleFactor(pyramid_scale_factor);
            pyramid_scale_factor_param.setModified(false);
        }
        cv::cuda::GpuMat dets;
        cv::cuda::GpuMat image = input->getGpuMat(&stream);
        m_gpu_detector->detectMultiScale(image, dets);
        std::vector<cv::Rect> det_rects;
        m_gpu_detector->convert(dets, det_rects);
        aq::CategorySet::ConstPtr labels = this->getLabels();
        aq::DetectedObjectSet detections(labels);

        auto size = input->size();
        for (cv::Rect2f rect : det_rects)
        {
            aq::DetectedObject det;
            det.classify((*labels)[0](1.0));
            aq::normalizeBoundingBox(rect, size);
            det.bounding_box = rect;
            detections.push_back(det);
        }

        this->detections.publish(std::move(detections));
        return true;
    }

} // namespace aqcore
