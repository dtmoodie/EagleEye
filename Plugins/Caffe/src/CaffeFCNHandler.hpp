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
                //PARAM(std::string, output_blob_name, "score")
                PARAM(float, min_confidence, 10)
            MO_END
            void HandleOutput(const caffe::Net<float>& net, boost::optional<mo::time_t> timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size, const std::vector<DetectedObject2d>& objs);
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
            virtual void HandleOutput(const caffe::Net<float>& net, boost::optional<mo::time_t> timestamp, const std::vector<cv::Rect>& bounding_boxes, cv::Size input_image_size, const std::vector<DetectedObject2d>& objs);
        };
    }
}
