#include "HaarDetector.hpp"
#if MO_OPENCV_HAVE_CUDA == 1
#include <Aquila/nodes/NodeContextSwitch.hpp>
#include <MetaObject/core/metaobject_config.hpp>
#include <boost/filesystem.hpp>

namespace aq
{
namespace nodes
{

bool HaarDetector::processImpl()
{
    if (use_gpu && MO_OPENCV_HAVE_CUDA == 0)
    {
        use_gpu = false;
    }
    MO_ASSERT(labels != nullptr);
    MO_ASSERT_GT(labels->size(), 0);
    return nodeContextSwitch(this, _ctx.get());
}

template <>
bool HaarDetector::processImpl(mo::Context* ctx)
{
    MO_ASSERT(use_gpu == false);
    if (!m_cpu_detector || model_file_param.modified())
    {
        if (!boost::filesystem::exists(model_file))
        {
            MO_LOG(warning) << "Cascade model file doesn't exist! " << model_file;
        }
        else
        {
            m_cpu_detector.reset(new cv::CascadeClassifier());
            m_cpu_detector->load(model_file.string());
            model_file_param.modified(false);
        }
    }
    if (m_cpu_detector)
    {
        cv::Mat img;
        img = input->getMat(ctx);
        auto original_window_size = m_cpu_detector->getOriginalWindowSize();
        auto regions = getRegions();
        for (const auto& region : regions)
        {
            cv::Mat roi = img(region);
            std::vector<cv::Rect> det_rects;
            m_cpu_detector->detectMultiScale(
                roi, det_rects, pyramid_scale_factor, min_neighbors, 0, min_object_size, max_object_size);
            detections.clear();
            auto size = input->getSize();
            for (cv::Rect2f rect : det_rects)
            {
                rect.x += region.x;
                rect.y += region.y;
                DetectedObject det;
                det.classify((*labels)[0](1.0));
                normalizeBoundingBox(rect, size);
                det.bounding_box = rect;
                detections.push_back(det);
            }
        }

        detections_param.emitUpdate(input_param);
        return true;
    }
    return false;
}
}
}
#endif