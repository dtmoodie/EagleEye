#include "EagleLib/nodes/Node.h"
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
    }
}
