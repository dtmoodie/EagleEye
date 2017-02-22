#pragma once
#include "CaffeNetHandler.hpp"
#include "EagleLib/ObjectDetection.hpp"
namespace EagleLib
{
    namespace Caffe
    {
        class SSDHandler: public NetHandler
        {
        public:
            static std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net);
            MO_DERIVE(SSDHandler, NetHandler)
                PARAM(std::vector<float>, detection_threshold, {0.75})
                OUTPUT(std::vector<DetectedObject>, detections, std::vector<DetectedObject>())
            MO_END
            virtual void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes);

        };

    }
}
