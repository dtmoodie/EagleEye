#include <boost/function.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace EagleLib
{
typedef boost::function<int(cv::cuda::GpuMat,cv::cuda::GpuMat,cv::cuda::Stream,int)> setReferenceFunctor;
typedef boost::function<cv::cuda::GpuMat(cv::cuda::GpuMat, cv::cuda::GpuMat*, cv::cuda::GpuMat*)> trackFunctor;
}

