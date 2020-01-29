#include "FaceRecognizer.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
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
    THROW(warning) << "Not actually implemented";
}
}

namespace aq
{
namespace nodes
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
        MO_LOG(warning) << "Failed to initialize face recognition model from landmark file " << shape_landmark_file;
        return false;
    }
    output.clear();
    if (detections->size())
    {
        cv::Mat img = image->getMat(_ctx.get());
        auto size = image->getSize();

        dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
        std::vector<dlib::matrix<dlib::bgr_pixel>> aligned_faces;
        {
            for (auto det : *detections)
            {
                if (min_size < 1.0F)
                {
                    if (det.bounding_box.width < min_size || det.bounding_box.height < min_size)
                    {
                        continue;
                    }
                }
                boundingBoxToPixels(det.bounding_box, size);

                if (min_size > 1.0F)
                {
                    if (det.bounding_box.width < min_size || det.bounding_box.height < min_size)
                    {
                        continue;
                    }
                }

                dlib::rectangle rect(det.bounding_box.x,
                                     det.bounding_box.y,
                                     det.bounding_box.x + det.bounding_box.width,
                                     det.bounding_box.y + det.bounding_box.height);
                dlib::full_object_detection shape = m_face_aligner(dlib_img, rect);

                LandmarkDetection aligned_det(det);
                aligned_det.bounding_box.x = shape.get_rect().left();
                aligned_det.bounding_box.y = shape.get_rect().top();
                aligned_det.bounding_box.width = shape.get_rect().width();
                aligned_det.bounding_box.height = shape.get_rect().height();
                const auto num_points = shape.num_parts();
                cv::Mat_<cv::Point2f> pts(1, num_points);
                for (size_t i = 0; i < num_points; ++i)
                {
                    const auto& part = shape.part(i);
                    pts(i).x = part.x();
                    pts(i).y = part.y();
                }
                aligned_det.landmark_keypoints = pts;
                output.push_back(std::move(aligned_det));
            }
        }
    }
    output_param.emitUpdate(detections_param);
    return true;
}

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
}
}
using namespace aq::nodes;
MO_REGISTER_CLASS(FaceRecognizer)
MO_REGISTER_CLASS(FaceAligner)
