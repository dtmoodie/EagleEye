#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/utilities/ColorMapping.hpp>
#include <MetaObject/params/TMultiInput-inl.hpp>
#include "DetectionFilter.hpp"

namespace aq
{
namespace nodes
{

class DrawDetections : public Node
{
  public:
    MO_DERIVE(DrawDetections, Node)
        INPUT(SyncedMemory, image, nullptr)
        MULTI_INPUT(detections, aq::DetectedObjectSet, aq::DetectionDescriptionSet, aq::DetectionDescriptionPatchSet)

        PARAM(bool, draw_class_label, true)
        PARAM(bool, draw_detection_id, true)
        PARAM(bool, publish_empty_dets, true)

        OUTPUT(SyncedMemory, output, SyncedMemory())
    MO_END
    template<class DType>
    void apply(bool* success);
  protected:
    void drawMetaData(cv::Mat& mat, const aq::DetectedObject& det, const cv::Rect2f& rect, const size_t idx);
    void drawMetaData(cv::Mat& mat, const aq::DetectionDescription& det, const cv::Rect2f& rect, const size_t idx);
    void drawMetaData(cv::Mat& mat, const aq::DetectionDescriptionPatch& det, const cv::Rect2f& rect, const size_t idx);

    virtual bool processImpl() override;
    std::string textLabel(const DetectedObject& det);
};

}
}
