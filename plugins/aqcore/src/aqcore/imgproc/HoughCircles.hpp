#pragma once
#include <aqcore_export.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedImage.hpp>
#include <Aquila/types/geometry/Circle.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include <aqcore/aqcore_export.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aqcore
{
    class aqcore_EXPORT HoughCircle : public aq::nodes::Node
    {
      public:
        using Components_t = ct::VariadicTypedef<aq::Circlef, aq::detection::Confidence>;
        using Output_t = aq::TEntityComponentSystem<Components_t>;

        MO_DERIVE(HoughCircle, aq::nodes::Node)
            INPUT(aq::SyncedImage, input)

            PARAM(int, min_radius, 0)
            PARAM(int, max_radius, 0)
            PARAM(double, dp, 1.0)
            PARAM(double, upper_threshold, 200.0)
            PARAM(double, center_threshold, 100.0)

            OUTPUT(Output_t, output)
            OUTPUT(aq::SyncedImage, drawn_circles)
        MO_END;

      protected:
        virtual bool processImpl() override;
    };
} // namespace aqcore
