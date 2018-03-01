#include "IClassifier.hpp"
#include <Aquila/types/ObjectDetection.hpp>
namespace aq
{
    namespace nodes
    {
        class IImageDetector : virtual public IClassifier
        {
          public:
            MO_DERIVE(IImageDetector, IClassifier)
                OUTPUT(DetectedObjectSet, detections, {})
            MO_END;
        };
    }
}
