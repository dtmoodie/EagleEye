#pragma once
#include "Aquila/types/ObjectDetection.hpp"
#include "CaffeNetHandler.hpp"
namespace aq
{
    namespace Caffe
    {
        class ClassifierHandler : public NetHandler
        {
          public:
            static std::map<int, int> CanHandleNetwork(const caffe::Net<float>& net);

            MO_DERIVE(ClassifierHandler, NetHandler)
            OUTPUT(std::vector<DetectedObject>, classified_detections, {})
            PARAM(float, classification_threshold, 0.5)
            MO_END
            virtual void startBatch();
            virtual void handleOutput(const caffe::Net<float>& net,
                                      const std::vector<cv::Rect>& bounding_boxes,
                                      mo::ITParam<aq::SyncedMemory>& input_param,
                                      const std::vector<DetectedObject2d>& objs);
            virtual void endBatch(boost::optional<mo::Time_t> timestamp);
        };
    }
}
