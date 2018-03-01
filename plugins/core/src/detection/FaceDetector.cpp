#include "FaceDetector.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

namespace aq
{
    namespace nodes
    {
        bool HaarFaceDetector::processImpl()
        {
            if (labels.size() == 0)
            {
                labels = aq::CategorySet(std::vector<std::string>({"face"}));
            }
            std::vector<cv::Rect> faces;
            if (input->getSyncState() <= aq::SyncedMemory::DEVICE_UPDATED)
            {
                if (!m_cpu_detector || model_file_param.modified())
                {
                    m_cpu_detector.reset(new cv::CascadeClassifier());
                    m_cpu_detector->load(model_file.string());
                }
                cv::Mat img = input->getMat(stream());
                stream().waitForCompletion();
                m_cpu_detector->detectMultiScale(
                    img, faces, pyramid_scale_factor, min_neighbors, 0, min_object_size, max_object_size);
            }
            else
            {
                if (!m_gpu_detector || model_file_param.modified())
                {
                    m_gpu_detector = cv::cuda::CascadeClassifier::create(model_file.string());
                    auto max_size = m_gpu_detector->getMaxObjectSize();
                    auto min_size = m_gpu_detector->getMinObjectSize();
                    max_object_size = max_size;
                    min_object_size = min_size;
                    pyramid_scale_factor = m_gpu_detector->getScaleFactor();
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
                m_gpu_detector->detectMultiScale(input->getGpuMat(stream()), dets);
                m_gpu_detector->convert(dets, faces);
            }

            detections.clear();
            for (auto& face : faces)
            {
                DetectedObject<5> det;
                det.bounding_box = face;
                detections.push_back(det);
            }
            detections_param.emitUpdate(input_param);
            return true;
        }
    }
}
using namespace aq::nodes;
MO_REGISTER_CLASS(HaarFaceDetector)
