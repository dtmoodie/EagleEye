#pragma once



#include <opencv2/core/cuda.hpp>


#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

void Depth2XYZ(cv::cuda::GpuMat depth, cv::cuda::GpuMat& XYZ, cv::cuda::Stream stream);