#include "HaarDetector.hpp"

#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <MetaObject/core/metaobject_config.hpp>
#include <boost/filesystem.hpp>

namespace aqcore
{

    bool HaarDetector::processImpl(mo::IAsyncStream& stream)
    {
        MO_ASSERT(use_gpu == false);
        if (!m_cpu_detector || model_file_param.getModified())
        {
            if (!boost::filesystem::exists(model_file))
            {
                this->getLogger().warn("Cascade model file doesn't exist! {}", model_file);
            }
            else
            {
                m_cpu_detector.reset(new cv::CascadeClassifier());
                m_cpu_detector->load(model_file.string());
                model_file_param.setModified(false);
            }
        }
        if (m_cpu_detector)
        {
            cv::Mat img;
            img = input->getMat(&stream);
            // auto original_window_size = m_cpu_detector->getOriginalWindowSize();
            auto regions = getRegions();
            aq::CategorySet::ConstPtr labels = this->getLabels();
            aq::DetectedObjectSet detections(labels);
            for (const auto& region : regions)
            {
                cv::Mat roi = img(region);
                std::vector<cv::Rect> det_rects;
                m_cpu_detector->detectMultiScale(
                    roi, det_rects, pyramid_scale_factor, min_neighbors, 0, min_object_size, max_object_size);

                auto size = input->size();
                for (cv::Rect2f rect : det_rects)
                {
                    rect.x += region.x;
                    rect.y += region.y;
                    aq::DetectedObject det;
                    det.classify((*labels)[0](1.0));
                    normalizeBoundingBox(rect, size);
                    det.bounding_box = rect;
                    detections.push_back(det);
                }
            }
            this->detections.publish(std::move(detections));
            return true;
        }
        return false;
    }

} // namespace aqcore
