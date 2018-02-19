#pragma once
#define CVAPI_EXPORTS

#include <opencv2/core/cuda.hpp>

namespace cv
{
    namespace cuda
    {
        double CV_EXPORTS kmeans(GpuMat samples, int K, GpuMat& labels, TermCriteria termCrit, int attempts, int flags, GpuMat& centers, Stream stream = Stream::Null(), GpuMat weights = GpuMat());
    }
}
