#pragma once
#include <string>
#include <opencv2/core/types.hpp>
#include <vector>
#include <EagleLib/Detail/Export.hpp>
namespace EagleLib
{
    struct EAGLE_EXPORTS Classification
    {
        Classification(const std::string& label_ = "", float confidence_ = 0, int classNumber_ = -1);
        std::string label;
        float confidence;
        int classNumber;
        template<class AR> void serialize(AR& ar);
    };
    
    struct EAGLE_EXPORTS DetectedObject
    {
        std::vector<Classification> detections;
        cv::Rect2f boundingBox;
        template<class AR> void serialize(AR& ar);
    };
}