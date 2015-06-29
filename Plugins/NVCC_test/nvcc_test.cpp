#include "nvcc_test.h"
#include "nvcc_test.cuh"
#include "RuntimeSourceDependency.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
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
    run_kernel(img.data, img.size().area()*img.cols, cv::cuda::StreamAccessor::getStream(stream));

    return img;
}

//ADD_RUNTIME_SOURCE_DEPENDENCY_ABS("/mnt/src/EagleEye/Plugins/NVCC_test/nvcc_test.cu")

ADD_RUNTIME_SOURCE_DEPENDENCY_REL("nvcc_test.cu")

NODE_DEFAULT_CONSTRUCTOR_IMPL(nvcc_test)

