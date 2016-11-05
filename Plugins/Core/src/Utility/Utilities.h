#include "EagleLib/Nodes/Node.h"
#include <MetaObject/MetaObject.hpp>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace EagleLib
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
                PARAM(cv::Rect, roi, cv::Rect(0,0,0,0));
                INPUT(SyncedMemory, image, nullptr);
                OUTPUT(SyncedMemory, ROI, SyncedMemory());
            MO_END
        protected:
            bool ProcessImpl();
        };
    }
}
