#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include "IClassifier.hpp"

#include <Aquila/types/ObjectDetection.hpp>

namespace aq
{
    namespace nodes
    {
        class IImageDetector : virtual public IClassifier
        {
          public:
            MO_DERIVE(IImageDetector, IClassifier)
                INPUT(SyncedImage, input)

                OPTIONAL_INPUT(std::vector<cv::Rect2f>, bounding_boxes)
                OPTIONAL_INPUT(DetectedObjectSet, input_detections)

                OUTPUT(DetectedObjectSet, detections)
            MO_END;

          protected:
            std::vector<cv::Rect> getRegions() const;
        };
    } // namespace nodes
} // namespace aq
