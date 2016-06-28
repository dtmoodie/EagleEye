#include <EagleLib/nodes/Node.h>
#include "EagleLib/Defs.hpp"
#include <EagleLib/Project_defs.hpp>
SETUP_PROJECT_DEF

namespace EagleLib
{
    namespace Nodes
    {
    class nvcc_test: public Node
    {
    public:
         
        nvcc_test();
        virtual void NodeInit(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
    }
}
