#pragma once
#include <string>
#include <opencv2/core/types.hpp>

struct Classification
{
    std::string label;
    float confidence;
    int classNumber;
};


struct DetectedObject
{
    std::vector<Classification> detections;
    cv::Rect boundingBox;
};
