#include "nvcc_test.h"
#include "nvcc_test.cuh"

using namespace EagleLib;

IPerModuleInterface* GetModule()
{
    return PerModuleInterface::GetInstance();
}
void SetupIncludes()
{
    EagleLib::NodeManager::getInstance().addIncludeDir("/mnt/src/EagleLib/Plugins/NVCC_test");
    EagleLib::NodeManager::getInstance().addSourceFile("/mnt/src/EagleLib/Plugins/NVCC_test/nvcc_test.cu");

}

void nvcc_test::init(bool firstInit)
{

}

cv::cuda::GpuMat nvcc_test::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    run_kernel();
    return img;
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(nvcc_test)
