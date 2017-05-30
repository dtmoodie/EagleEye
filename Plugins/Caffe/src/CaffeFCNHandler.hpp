#pragma once
#include "CaffeNetHandler.hpp"

namespace aq
{
    namespace Caffe
    {
        class FCNHandler: public NetHandler
        {
        public:
            static std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net);
            MO_DERIVE(FCNHandler, NetHandler)
                OUTPUT(SyncedMemory, label, SyncedMemory())
                OUTPUT(SyncedMemory, confidence, SyncedMemory())
                PARAM(float, min_confidence, 10)
            MO_END
            virtual void handleOutput(const caffe::Net<float>& net, const std::vector<cv::Rect>& bounding_boxes, mo::ITParam<aq::SyncedMemory>& input_param, const std::vector<DetectedObject2d>& objs);
        };
        class FCNSingleClassHandler: public NetHandler
        {
        public:
            static std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net);
            MO_DERIVE(FCNSingleClassHandler, NetHandler)
                OUTPUT(SyncedMemory, label, SyncedMemory())
                OUTPUT(SyncedMemory, confidence, SyncedMemory())

                PARAM(float, min_confidence, 10)
                PARAM(int, class_index, 0)
            MO_END
            virtual void handleOutput(const caffe::Net<float>& net, const std::vector<cv::Rect>& bounding_boxes, mo::ITParam<aq::SyncedMemory>& input_param, const std::vector<DetectedObject2d>& objs);
        };
    }
}
