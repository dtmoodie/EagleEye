#pragma once
#include <Aquila/types/SyncedImage.hpp>

#include "IClassifier.hpp"

#include <Aquila/types/ObjectDetection.hpp>

namespace aqcore
{

    class IImageDetector : virtual public IClassifier
    {
      public:
        using OutputComponents_t =
            ct::VariadicTypedef<aq::detection::BoundingBox2d, aq::detection::Confidence, aq::detection::Id>;
        using Output_t = aq::TDetectedObjectSet<OutputComponents_t>;

        using Input_t = aq::TDetectedObjectSet<ct::VariadicTypedef<aq::detection::BoundingBox2d>>;

        MO_DERIVE(IImageDetector, IClassifier)
            INPUT(aq::SyncedImage, input)

            OPTIONAL_INPUT(std::vector<cv::Rect2f>, regions_of_interest)
            OPTIONAL_INPUT(Input_t, input_detections)

            OUTPUT(Output_t, detections)
        MO_END;

      protected:
        std::vector<cv::Rect> getRegions() const;
    };

} // namespace aqcore
