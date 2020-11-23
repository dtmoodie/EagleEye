#include <ct/types/opencv.hpp>
#include <dlib/opencv.h>

#include "FaceAligner.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

namespace aqdlib
{
    bool FaceAligner::processImpl()
    {
        if (!m_initialized && boost::filesystem::exists(shape_landmark_file))
        {
            dlib::deserialize(shape_landmark_file.string()) >> m_face_aligner;
            m_initialized = true;
        }
        if (!m_initialized)
        {
            this->getLogger().warn("Failed to initialize face alignment model from landmark file {}",
                                   shape_landmark_file);
            return false;
        }

        const uint32_t num_entities = detections->getNumEntities();

        if (num_entities > 0)
        {
            mo::IAsyncStreamPtr_t stream = this->getStream();
            cv::Mat img = image->getMat(stream.get());
            const auto size = image->size();
            aq::TDetectedObjectSet<OutputComponents_t> out = *detections;

            dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
            std::vector<dlib::matrix<dlib::bgr_pixel>> aligned_faces;
            {
                mt::Tensor<const aq::detection::BoundingBox2d::DType, 1> bbs =
                    detections->getComponent<aq::detection::BoundingBox2d>();

                mt::Tensor<aq::detection::BoundingBox2d::DType, 1> out_bbs =
                    out.getComponentMutable<aq::detection::BoundingBox2d>();

                mt::Tensor<cv::Point2f, 2> landmarks;

                for (uint32_t i = 0; i < num_entities; ++i)
                {
                    cv::Rect2f bb = bbs[i];
                    if (min_size < 1.0F)
                    {
                        if (bb.width < min_size || bb.height < min_size)
                        {
                            continue;
                        }
                    }

                    aq::boundingBoxToPixels(bb, size);

                    if (min_size > 1.0F)
                    {
                        if (bb.width < min_size || bb.height < min_size)
                        {
                            continue;
                        }
                    }

                    dlib::rectangle rect(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);
                    dlib::full_object_detection shape = m_face_aligner(dlib_img, rect);

                    cv::Rect2f& out_bb = out_bbs[i];

                    out_bb.x = shape.get_rect().left();
                    out_bb.y = shape.get_rect().top();
                    out_bb.width = shape.get_rect().width();
                    out_bb.height = shape.get_rect().height();

                    const auto num_points = shape.num_parts();
                    if (landmarks.getShape()[0] == 0)
                    {
                        out.reshape<aq::detection::LandmarkDetection>(mt::Shape<2>{num_entities, num_points});
                        landmarks = out.getComponentMutable<aq::detection::LandmarkDetection>();
                    }

                    for (size_t j = 0; j < num_points; ++j)
                    {
                        const auto& part = shape.part(j);
                        landmarks(i, j).x = part.x();
                        landmarks(i, j).y = part.y();
                    }
                }
            }
            this->output.publish(std::move(out), mo::tags::param = &this->detections_param);
        }

        return true;
    }
} // namespace aqdlib

using namespace aqdlib;
MO_REGISTER_CLASS(FaceAligner)