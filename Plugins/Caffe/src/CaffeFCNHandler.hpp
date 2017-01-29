#pragma once
#include "CaffeNetHandler.hpp"

namespace EagleLib
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
                //PARAM(std::string, output_blob_name, "score")
                PARAM(float, min_confidence, 10)
            MO_END
            virtual void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes);
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
            virtual void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes);
        };
        class LaneHandler: public NetHandler
        {
        public:
            static std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net);
            MO_DERIVE(LaneHandler, NetHandler)
                OUTPUT(SyncedMemory, lane, SyncedMemory())
                OUTPUT(SyncedMemory, confidence, SyncedMemory())
                PARAM(float, min_confidence, 10)
                PARAM(int, pad_x, 14)
                PARAM(int, pad_y, 23)
                PARAM(int, size_x, 254)
                PARAM(int, size_y, 158)
                PARAM(int, rf_x, 68)
                PARAM(int, rf_y, 32)
            MO_END
            virtual void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes);
        };
    }
}
