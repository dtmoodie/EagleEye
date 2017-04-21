#pragma once
#include "CaffeNetHandler.hpp"
#include "Aquila/ObjectDetection.hpp"
namespace aq
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
                STATUS(int, num_detections, 0)
            MO_END
            virtual void StartBatch(){ current_id = 0;}
            void HandleOutput(const caffe::Net<float>& net, boost::optional<mo::time_t> timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size, const std::vector<DetectedObject2d>& objs);
            int current_id;
        };

    }
}
