#include "Aquila/Nodes/Node.h"
#include <MetaObject/MetaObject.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace Nodes
    {
        class ApplyEveryNFrames: public Node
        {
        public:
            ApplyEveryNFrames();
        };
        class SyncFunctionCall: public Node
        {
            int numInputs = 0;
        public:
            void call();
            SyncFunctionCall();
            virtual void NodeInit(bool firstInit);
            virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
        };
        class SyncBool: public Node
        {

        };
        class RegionOfInterest : public Node
        {
        public:
            MO_DERIVE(RegionOfInterest, Node)
                PARAM(cv::Rect2f, roi, cv::Rect2f(0.0f,0.0f,1.0f,1.0f))
                INPUT(SyncedMemory, image, nullptr)
                OUTPUT(SyncedMemory, ROI, SyncedMemory())
            MO_END
        protected:
            bool ProcessImpl();
        };
        class ExportRegionsOfInterest: public Node
        {
        public:
            MO_DERIVE(ExportRegionsOfInterest, Node)
                PARAM(std::vector<cv::Rect2f>, rois, {})
            MO_END
            mo::TypedParameterPtr<std::vector<cv::Rect2f>> output;
            void NodeInit(bool firstInit);
        protected:
            bool ProcessImpl();

        };
    }
}
