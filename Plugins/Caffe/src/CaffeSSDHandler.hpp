#pragma once
#include "CaffeNetHandler.hpp"
#include "Aquila/types/ObjectDetection.hpp"
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
            virtual void startBatch(){ current_id = 0;}
            virtual void handleOutput(const caffe::Net<float>& net, const std::vector<cv::Rect>& bounding_boxes,
                                      mo::ITParam<aq::SyncedMemory>& input_param, const std::vector<DetectedObject2d>& objs);
            int current_id;
        };

    }
}
