#include "FaceDetector.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <boost/filesystem.hpp>

#include <dlib/opencv.h>
namespace aqdlib
{

    bool DlibMMODDetector::processImpl(mo::IAsyncStream& stream)
    {
        if (!m_net || model_file_param.getModified())
        {
            void createLabels();
            m_net.reset(new dlib::mmod_net_type());
            if (boost::filesystem::exists(model_file))
            {
                dlib::deserialize(model_file.string()) >> *m_net;
                model_file_param.setModified(false);
            }
            else
            {
                this->getLogger().error("Unable to load non-existant face detector file {}", model_file.string());
                return false;
            }
        }

        cv::Mat img = input->getMat(&stream);

        std::vector<dlib::cv_image<dlib::bgr_pixel>> dlib_img{dlib::cv_image<dlib::bgr_pixel>(img)};

        std::vector<std::vector<dlib::mmod_rect>> dets = (*m_net)(dlib_img);
        Output_t output;
        output.resize(dets[0].size());

        for (size_t i = 0; i < dets[0].size(); ++i)
        {
            const dlib::mmod_rect& det = dets[0][i];
            output[i] =
                aq::detection::BoundingBox2d(det.rect.left(), det.rect.top(), det.rect.width(), det.rect.height());
            output[i] = aq::detection::Confidence(det.detection_confidence);
        }
        this->output.publish(std::move(output), mo::tags::param = &input_param);
        return true;
    }

    bool DlibMMODDetector::processImpl(mo::IDeviceStream& stream) { return false; }

} // namespace aqdlib

using namespace aqdlib;
MO_REGISTER_CLASS(DlibMMODDetector);
