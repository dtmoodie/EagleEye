#pragma once
#include "CaffeNetHandler.hpp"

namespace EagleLib
{
    namespace Caffe
    {
        class FCNHandler: public NetHandler
        {
        public:
            static std::vector<int> CanHandleNetwork(const caffe::Net<float>& net);
            MO_DERIVE(FCNHandler, NetHandler)
                OUTPUT(SyncedMemory, label, SyncedMemory())
                OUTPUT(SyncedMemory, confidence, SyncedMemory())
                PARAM(std::string, output_blob_name, "score")
                PARAM(float, min_confidence, 10)
            MO_END
            virtual void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes);
        protected:
        };

    }
}
