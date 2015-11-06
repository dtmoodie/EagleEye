#include <nodes/Node.h>
#include "EagleLib/Defs.hpp"
SETUP_PROJECT_DEF

namespace EagleLib
{
    class nvcc_test: public Node
    {
    public:
		 
        nvcc_test();
        virtual void init(bool firstInit);
        virtual cv::cuda::GpuMat doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream);
    };
}
