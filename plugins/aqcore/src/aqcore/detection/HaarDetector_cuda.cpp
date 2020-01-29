#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include "HaarDetector.hpp"
#include <boost/filesystem.hpp>
#include <MetaObject/core/CvContext.hpp>

namespace aq
{
namespace nodes
{

template<>
bool HaarDetector::processImpl(mo::CvContext* ctx)
{

    if(!use_gpu)
    {
        bool sync = false;
        input->getMat(ctx, 0, &sync);
        if(sync)
        {
            ctx->getStream().waitForCompletion();
        }
        return processImpl(static_cast<mo::Context*>(ctx));
    }
    if(!m_gpu_detector || model_file_param.modified())
    {
        m_gpu_detector = cv::cuda::CascadeClassifier::create(model_file.string());
        model_file_param.modified(false);
    }
    if (max_object_size_param.modified())
    {
        m_gpu_detector->setMaxObjectSize(max_object_size);
        max_object_size_param.modified(false);
    }
    if (min_object_size_param.modified())
    {
        m_gpu_detector->setMinObjectSize(min_object_size);
        min_object_size_param.modified(false);
    }
    if (pyramid_scale_factor_param.modified())
    {
        m_gpu_detector->setScaleFactor(pyramid_scale_factor);
        pyramid_scale_factor_param.modified(false);
    }
    cv::cuda::GpuMat dets;
    ctx->getStream().waitForCompletion(); // need to fix opencv's implementation, cannot handle streams :/
    m_gpu_detector->detectMultiScale(input->getGpuMat(ctx), dets);
    std::vector<cv::Rect> det_rects;
    m_gpu_detector->convert(dets, det_rects);
    detections.clear();
    auto size = input->getSize();
    for (cv::Rect2f rect : det_rects)
    {
        DetectedObject det;
        det.classify((*labels)[0](1.0));
        normalizeBoundingBox(rect, size);
        det.bounding_box = rect;
        detections.push_back(det);
    }

    detections_param.emitUpdate(input_param);
    return true;
}

}
}
#endif
