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
                PARAM(mo::ReadFile, label_file, {})
                PARAM(std::vector<std::string>, labels, {})
            MO_END

            void HandleOutput(const caffe::Net<float>& net, long long timestamp, const std::vector<cv::Rect>& bounding_boxes);
        };
    }
}
