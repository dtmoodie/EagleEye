#include "EagleLib/ObjectDetection.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
//#include "MetaObject/Parameters/IO/TextPolicy.hpp"
using namespace EagleLib;

EagleLib::Classification::Classification(const std::string& label_, float confidence_, int classNumber_) :
    label(label_), confidence(confidence_), classNumber(classNumber_) 
{

}

namespace cereal
{
    template<class AR, class T> void serialize(AR& ar, cv::Rect_<T>& rect)
    {
        ar(make_nvp("x", rect.x), make_nvp("y", rect.y), make_nvp("width", rect.width), make_nvp("height", rect.height));
    }
}

template<class AR> void Classification::serialize(AR& ar)
{
    ar(CEREAL_NVP(label), CEREAL_NVP(confidence), CEREAL_NVP(classNumber));
}

template<class AR> void DetectedObject::serialize(AR& ar)
{
    ar(CEREAL_NVP(boundingBox), CEREAL_NVP(detections));
}

INSTANTIATE_META_PARAMETER(DetectedObject);
INSTANTIATE_META_PARAMETER(Classification);
INSTANTIATE_META_PARAMETER(std::vector<DetectedObject>);