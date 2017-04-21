#pragma once
#include "CaffeNetHandler.hpp"
#include "Aquila/ObjectDetection.hpp"
namespace aq
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
            void HandleOutput(const caffe::Net<float>& net, boost::optional<mo::time_t> timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size, const std::vector<DetectedObject2d>& objs);
            void EndBatch(boost::optional<mo::time_t> timestamp);
        };
    }
}
