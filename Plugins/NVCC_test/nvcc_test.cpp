#include "nvcc_test.h"
#include "nvcc_test.cuh"
#include "RuntimeSourceDependency.h"
#include <opencv2/core/cuda_stream_accessor.hpp>


using namespace EagleLib;

void SetupIncludes()  
{
#ifdef NVCC_TEST_INCLUDE
	EagleLib::NodeManager::getInstance().addIncludeDir(NVCC_TEST_INCLUDE);
#endif
}

void nvcc_test::init(bool firstInit)  
{

}

cv::cuda::GpuMat nvcc_test::doProcess(cv::cuda::GpuMat &img, cv::cuda::Stream &stream)
{
    run_kernel(img.data, img.size().area()*img.cols, cv::cuda::StreamAccessor::getStream(stream));

    return img;
} 


NODE_DEFAULT_CONSTRUCTOR_IMPL(nvcc_test)  

