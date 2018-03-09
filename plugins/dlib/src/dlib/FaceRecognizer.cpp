#include "FaceRecognizer.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <boost/filesystem.hpp>

#include <dlib/image_processing.h>
#include <dlib/opencv.h>

namespace dlib
{
    void set_image_size(cv_image<bgr_pixel>& image, const long unsigned int& rows, const long unsigned int& cols)
    {
        THROW(warning) << "Not actually implemented";
    }
}

namespace aq
{
    namespace nodes
    {
        bool FaceRecognizer::processImpl()
        {
            if (!m_initialized && boost::filesystem::exists(shape_landmark_file))
            {
                dlib::deserialize(shape_landmark_file.string()) >> m_face_aligner;
                dlib::deserialize(face_recognizer_weight_file.string()) >> m_net;
                m_initialized = true;
            }
            if (!m_initialized)
            {
                return false;
            }
            if (detections->size())
            {
                const cv::Mat& img = image->getMat(stream());
                stream().waitForCompletion();
                dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
                std::vector<dlib::matrix<dlib::bgr_pixel>> aligned_faces;
                for (const auto& det : *detections)
                {
                    dlib::rectangle rect(det.bounding_box.x,
                                         det.bounding_box.y,
                                         det.bounding_box.x + det.bounding_box.width,
                                         det.bounding_box.y + det.bounding_box.height);
                    auto shape = m_face_aligner(dlib_img, rect);
                    cv::Mat roi(150, 150, CV_8UC3);
                    dlib::matrix<dlib::bgr_pixel> face_chip;
                    dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
                }
                std::vector<dlib::matrix<float, 0, 1>> face_descriptors = m_net(aligned_faces);
            }

            return true;
        }
    }
}
using namespace aq::nodes;
MO_REGISTER_CLASS(FaceRecognizer)
