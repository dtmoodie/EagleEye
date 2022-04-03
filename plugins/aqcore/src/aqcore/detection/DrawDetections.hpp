#pragma once
#include "../OpenCVCudaNode.hpp"

#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/ObjectDetection.hpp>

#include <Aquila/types/SyncedImage.hpp>

#include "DetectionFilter.hpp"

namespace aqcore
{

    class DrawDetections : public OpenCVCudaNode
    {
      public:
        using BoundingBox2d = aq::detection::BoundingBox2d;
        using Classifications = aq::detection::Classifications;
        using Confidence = aq::detection::Confidence;
        using Id = aq::detection::Id;
        using Pose3d = aq::detection::Pose3d;
        using Size3d = aq::detection::Size3d;
        using Descriptor = aq::detection::Descriptor;

        MO_DERIVE(DrawDetections, OpenCVCudaNode)
            INPUT(aq::SyncedImage, image)
            INPUT(aq::DetectedObjectSet, detections)

            PARAM(bool, draw_class_label, true)
            PARAM(bool, draw_detection_id, true)
            PARAM(bool, publish_empty_dets, true)

            OUTPUT(aq::SyncedImage, output)
        MO_END;

      protected:
        void drawBoxes(cv::Mat& mat,
                       mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                       mt::Tensor<const Classifications, 1> cls);

        void drawBoxes(cv::cuda::GpuMat& mat,
                       mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                       mt::Tensor<const Classifications, 1> cls,
                       cv::cuda::Stream& stream);

        void drawLabels(cv::Mat& mat,
                        mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                        mt::Tensor<const Classifications, 1> cats,
                        mt::Tensor<const Id::DType, 1> ids);

        void drawLabels(cv::cuda::GpuMat& mat,
                        mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                        mt::Tensor<const Classifications, 1> cats,
                        mt::Tensor<const Id::DType, 1> ids,
                        cv::cuda::Stream& stream);

        void drawDescriptors(cv::Mat3b mat,
                             mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                             mt::Tensor<const float, 2> descriptors);

        void drawDescriptors(cv::cuda::GpuMat& mat,
                             mt::Tensor<const BoundingBox2d::DType, 1> bbs,
                             mt::Tensor<const float, 2> descriptors,
                             cv::cuda::Stream& stream);

        bool processImpl() override;
        std::string textLabel(const Classifications& cats, const Id& id);
        cv::Mat textImage(const Classifications& cats, const Id& id);
    };

} // namespace aqcore
