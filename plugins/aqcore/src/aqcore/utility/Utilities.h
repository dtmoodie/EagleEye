#include <Aquila/types/SyncedImage.hpp>

#include <Aquila/nodes/Node.hpp>

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include <MetaObject/object/MetaObject.hpp>

RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace nodes
    {
        class ApplyEveryNFrames : public Node
        {
          public:
            ApplyEveryNFrames();
        };

        class SyncFunctionCall : public Node
        {
            int numInputs = 0;

          public:
            void call();
            SyncFunctionCall();
            virtual void nodeInit(bool firstInit);
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
        };

        class SyncBool : public Node
        {
        };

        class RegionOfInterest : public Node
        {
          public:
            MO_DERIVE(RegionOfInterest, Node)
                PARAM(cv::Rect2f, roi, cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f))
                INPUT(SyncedImage, image)
                OUTPUT(SyncedImage, ROI)
            MO_END;

          protected:
            bool processImpl();
        };
        class ExportRegionsOfInterest : public Node
        {
          public:
            MO_DERIVE(ExportRegionsOfInterest, Node)
                PARAM(std::vector<cv::Rect2f>, rois, {})

                OUTPUT(std::vector<cv::Rect2f>, output)
            MO_END;

            // mo::TParamPtr<std::vector<cv::Rect2f>> output;
            void nodeInit(bool firstInit);

          protected:
            bool processImpl();
        };
    } // namespace nodes
} // namespace aq
