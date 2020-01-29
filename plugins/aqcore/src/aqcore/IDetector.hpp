#pragma once
#include "IClassifier.hpp"
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>
namespace aq
{
namespace nodes
{
class IImageDetector : virtual public IClassifier
{
  public:
    MO_DERIVE(IImageDetector, IClassifier)
        INPUT(SyncedMemory, input, nullptr)

        OPTIONAL_INPUT(std::vector<cv::Rect2f>, bounding_boxes, nullptr)
        OPTIONAL_INPUT(DetectedObjectSet, input_detections, nullptr)

        OUTPUT(DetectedObjectSet, detections, {})
    MO_END;

  protected:
    std::vector<cv::Rect> getRegions() const;
};
}
}
