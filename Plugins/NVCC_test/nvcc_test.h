#include <nodes/Node.h>

#ifdef __cplusplus
extern "C"{
#endif
    IPerModuleInterface* GetModule();
    CV_EXPORTS void SetupIncludes();
#ifdef __cplusplus
}
#endif

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
