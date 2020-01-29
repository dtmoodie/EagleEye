#include "FaceDetector.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <boost/filesystem.hpp>

#include <dlib/opencv.h>
namespace aq
{
namespace nodes
{

bool DlibMMODDetector::processImpl()
{
    if (!m_net || model_file_param.modified())
    {
        void createLabels();
        m_net.reset(new dlib::mmod_net_type());
        if (boost::filesystem::exists(model_file))
        {
            dlib::deserialize(model_file.string()) >> *m_net;
            model_file_param.modified(false);
        }
    }
    cv::Mat img;
    bool sync = false;
    img = input->getMat(_ctx.get(), 0);
    if (sync)
    {
        _ctx->getStream().waitForCompletion();
    }
    std::vector<dlib::cv_image<dlib::bgr_pixel>> dlib_img{dlib::cv_image<dlib::bgr_pixel>(img)};
    std::vector<std::vector<dlib::mmod_rect>> dets = (*m_net)(dlib_img);
    detections.clear();
    for (const dlib::mmod_rect& det : dets[0])
    {
        aq::DetectedObject aqdet;
        aqdet.bounding_box.x = det.rect.left();
        aqdet.bounding_box.y = det.rect.top();
        aqdet.bounding_box.width = det.rect.width();
        aqdet.bounding_box.height = det.rect.height();
        aqdet.confidence = static_cast<float>(det.detection_confidence);
        aqdet.classifications = (*labels)[0](1.0);
        detections.emplace_back(std::move(aqdet));
    }
    detections_param.emitUpdate(input_param);
    return true;
}

} // namespace aq::nodes
} // namespace aq

using namespace aq::nodes;
MO_REGISTER_CLASS(DlibMMODDetector);
