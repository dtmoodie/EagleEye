#include "nodes/Node.h"

namespace EagleLib
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
        virtual void Init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    class SyncBool: public Node
    {

    };
}
