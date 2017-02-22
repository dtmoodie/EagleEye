#pragma once
#include "CaffeNetHandler.hpp"
#include "EagleLib/ObjectDetection.hpp"
namespace EagleLib
{
    namespace Caffe
    {
        class ClassifierHandler: public NetHandler
        {
        public:
            static std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net);

            MO_DERIVE(ClassifierHandler, NetHandler)
                OUTPUT(std::vector<DetectedObject>, objects, {})
                PARAM(float, classification_threshold, 0.5)
            MO_END
            void StartBatch();
            void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size);
            void EndBatch(long long timestamp);
        };
    }
}
