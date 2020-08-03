#ifndef DETECTIONFILTER_HPP
#define DETECTIONFILTER_HPP

#include <Aquila/types/DetectionDescription.hpp>

#include <Aquila/nodes/Node.hpp>

#include <MetaObject/params/TMultiPublisher.hpp>
#include <MetaObject/params/TMultiSubscriber.hpp>

#define MULTI_OUTPUT(NAME, ...) mo::TMultiOutput<__VA_ARGS__> NAME##_param;

namespace aq
{
    namespace nodes
    {

        class DetectionFilter : public Node
        {
          public:
            MO_DERIVE(DetectionFilter, Node)
                INPUT(aq::DetectedObjectSet, input)
                PARAM(std::vector<std::string>, allow_classes, {})
                PARAM(std::vector<std::string>, reject_classes, {})
                OUTPUT(aq::DetectedObjectSet, output)
            MO_END;

          protected:
            bool processImpl() override;
        };
    } // namespace nodes
} // namespace aq

#endif // DETECTIONFILTER_HPP
