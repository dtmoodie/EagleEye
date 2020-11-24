#include <ct/types/opencv.hpp>

#include "FaceRecognizer.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/logging_macros.hpp>
#include <MetaObject/logging/profiling.hpp>

#include <boost/filesystem.hpp>

#include <cudnn.h>
#include <dlib/dnn/cudnn_dlibapi.h>
#include <dlib/dnn/tensor_tools.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

namespace dlib
{
    void set_image_size(cv_image<bgr_pixel>& image, const long unsigned int& rows, const long unsigned int& cols)
    {
        THROW(warn, "Not actually implemented");
    }
} // namespace dlib

namespace aqdlib
{

    bool FaceRecognizer::processImpl()
    {
        if (!m_initialized && boost::filesystem::exists(face_recognizer_weight_file))
        {
            dlib::deserialize(face_recognizer_weight_file.string()) >> m_net;
            m_initialized = true;
        }
        if (!m_initialized)
        {
            this->getLogger().warn("Failed to initialize face recognition model from landmark file {}",
                                   face_recognizer_weight_file);
            return false;
        }

        auto stream = this->getStream();

        const uint32_t num_entities = this->detections->getNumEntities();
        if (num_entities > 0)
        {
            cv::Mat img = image->getMat(stream.get());
            dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
            std::vector<dlib::matrix<dlib::bgr_pixel>> aligned_faces;

            const auto size = image->size();
            auto bbs = detections->getComponent<aq::detection::BoundingBox2d>();
            auto landmarks = detections->getComponent<aq::detection::LandmarkDetection>();
            if (landmarks.getShape()[0] == 0)
            {
                return false;
            }
            
            for (uint32_t i = 0; i < num_entities; ++i)
            {
                auto bb = bbs[i];
                aq::boundingBoxToPixels(bb, size);
                std::vector<dlib::point> parts;
                auto pts = landmarks[i];

                for (int j = 0; j < pts.getShape().numElements(); ++j)
                {
                    parts.emplace_back(dlib::point(pts[j].x, pts[j].y));
                }

                dlib::rectangle rect(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);

                dlib::full_object_detection shape(rect, parts);
                dlib::matrix<dlib::bgr_pixel> face_chip;
                auto chip_details = dlib::get_face_chip_details(shape, 150, 0.25);
                dlib::extract_image_chip(dlib_img, chip_details, face_chip);
                
                aligned_faces.emplace_back(std::move(face_chip));
            }

            if (!aligned_faces.empty())
            {
                // TODO figure out how to pass in the output, thus avoiding any need to copy data.
                std::vector<dlib::matrix<float, 0, 1>> face_descriptors = m_net(aligned_faces);
                aq::TDetectedObjectSet<OutputComponents_t> output = *this->detections;
                output.reshape<aq::detection::Descriptor>(
                    mt::Shape<2>(face_descriptors.size(), face_descriptors[0].nr()));
                auto descriptors = output.getComponentMutable<aq::detection::Descriptor>();
                //auto provider = output.getProvider<aq::detection::Descriptor>();
                output.reshape<aq::detection::AlignedPatch>(mt::Shape<1>(num_entities));
                auto aligned_patch = output.getComponentMutable<aq::detection::AlignedPatch>();
                auto stream = this->getStream();

                for (size_t i = 0; i < face_descriptors.size(); ++i)
                {
                    const float* start = face_descriptors[i].begin();
                    const float* end = face_descriptors[i].end();
                    ct::TArrayView<const float> view(start, end - start);

                    ct::TArrayView<float> dest = descriptors[i];
                    view.copyTo(dest);
                    std::shared_ptr<dlib::matrix<dlib::bgr_pixel>> matrix = std::make_shared<dlib::matrix<dlib::bgr_pixel>>(std::move(aligned_faces[i]));
                    const aq::Shape<2> shape(matrix->nr(), matrix->nc());
                    dlib::bgr_pixel* data = matrix->begin();
                    aq::BGR<uint8_t>* pixel_ptr = ct::ptrCast<aq::BGR<uint8_t>>(data);
                    aq::SyncedImage aligned(shape, pixel_ptr, std::move(matrix), stream);
                    aligned_patch[i] = aq::detection::AlignedPatch{*image, std::move(aligned)};
                }
                this->output.publish(output, mo::tags::param=&this->detections_param);
            }
        }
        return true;
    }

} // namespace aqdlib

using namespace aqdlib;
MO_REGISTER_CLASS(FaceRecognizer)
