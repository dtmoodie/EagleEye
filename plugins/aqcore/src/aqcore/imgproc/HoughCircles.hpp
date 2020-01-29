#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/geometry/Circle.hpp>
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
#include <aqcore/aqcore_export.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace nodes
    {
        class aqcore_EXPORT HoughCircle : public Node
        {
        public:
            MO_DERIVE(HoughCircle, Node)
                INPUT(SyncedMemory, input, nullptr)
                PARAM(int, min_radius, 0)
                PARAM(int, max_radius, 0)
                PARAM(double, dp, 1.0)
                PARAM(double, upper_threshold, 200.0)
                PARAM(double, center_threshold, 100.0)
                OUTPUT(std::vector<Detection<Circle<float>>>, circles, {})
                OUTPUT(SyncedMemory, drawn_circles, {})
                MO_END;

        protected:
            virtual bool processImpl() override;
        };

    }
}
