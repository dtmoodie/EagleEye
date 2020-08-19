#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include <MetaObject/object/MetaObject.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aqcore
{
    class RegionOfInterest : public aq::nodes::Node
    {
      public:
        MO_DERIVE(RegionOfInterest, aq::nodes::Node)
            PARAM(cv::Rect2f, roi, cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f))

            INPUT(aq::SyncedImage, input)
            OUTPUT(aq::SyncedImage, output)
        MO_END;

      protected:
        bool processImpl() override;
    };

    class ExportRegionsOfInterest : public aq::nodes::Node
    {
      public:
        MO_DERIVE(ExportRegionsOfInterest, aq::nodes::Node)
            PARAM(std::vector<cv::Rect2f>, rois, {})

            OUTPUT(std::vector<cv::Rect2f>, output)
        MO_END;

        void nodeInit(bool firstInit) override;

      protected:
        bool processImpl() override;
    };

} // namespace aqcore
