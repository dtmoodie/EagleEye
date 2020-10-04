#include <ct/types/opencv.hpp>

#include "FaceRecognizer.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/logging/logging.hpp>
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
        MO_THROW(warn, "Not actually implemented");
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
            MO_LOG(warning) << "Failed to initialize face recognition model from landmark file "
                            << face_recognizer_weight_file;
            return false;
        }
        output.clear();
        if (detections->size())
        {
            cv::Mat img = image->getMat(_ctx.get());
            dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
            std::vector<dlib::matrix<dlib::bgr_pixel>> aligned_faces;

            auto size = image->getSize();
            for (const auto& det : *detections)
            {
                auto bb = det.bounding_box;
                boundingBoxToPixels(bb, size);
                std::vector<dlib::point> parts;
                cv::Mat pts = det.landmark_keypoints.getMat(_ctx.get());
                for (int i = 0; i < pts.cols; ++i)
                {
                    parts.emplace_back(dlib::point(pts.at<cv::Point2f>(0, i).x, pts.at<cv::Point2f>(0, i).y));
                }

                dlib::rectangle rect(bb.x, bb.y, bb.x + bb.width, bb.y + bb.height);

                dlib::full_object_detection shape(rect, parts);
                dlib::matrix<dlib::bgr_pixel> face_chip;
                dlib::extract_image_chip(dlib_img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
                aligned_faces.emplace_back(std::move(face_chip));
            }

            if (!aligned_faces.empty())
            {
                std::vector<dlib::matrix<float, 0, 1>> face_descriptors = m_net(aligned_faces);
                for (size_t i = 0; i < face_descriptors.size(); ++i)
                {
                    float* start = face_descriptors[i].begin();
                    float* end = face_descriptors[i].end();
                    cv::Mat wrapped(1, end - start, CV_32F, start);
                    DetectionDescriptionPatch det;
                    det.bounding_box = (*detections)[i].bounding_box;
                    det.id = (*detections)[i].id;
                    det.descriptor = wrapped.clone();
                    det.classifications = (*detections)[i].classifications;
                    det.aligned_patch = dlib::toMat(aligned_faces[i]).clone();
                    det.confidence = (*detections)[i].confidence;
                    output.emplace_back(std::move(det));
                }
            }
        }
        output_param.emitUpdate(image_param);
        return true;
    }

} // namespace aqdlib

using namespace aqdlib;
MO_REGISTER_CLASS(FaceRecognizer)
