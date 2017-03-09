
#include <Aquila/Thrust_interop.hpp>
#include <vector_types.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda/utility.hpp>
#include <opencv2/core/cuda/type_traits.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>


template<class T>
void __global__ apply_lut_kernel(const cv::Vec3b* lut, const cv::cuda::PtrStepSz<T> in, cv::cuda::PtrStepSz<cv::Vec3b> out, int num_labels)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < in.cols && y < in.rows)
    {
        T idx = in(y,x);
        if(idx >= 0 && idx < num_labels)
            out(y,x) = lut[idx];
    }
}

namespace aq
{
    void applyColormap(const cv::cuda::GpuMat& input_8U,
                                 cv::cuda::GpuMat& output_8UC3,
                                 const cv::cuda::GpuMat& colormap,
                                 cv::cuda::Stream& stream)
    {
        CV_Assert(input_8U.size().area());
        CV_Assert(input_8U.depth() == CV_8U || input_8U.depth() == CV_32S);
        CV_Assert(input_8U.channels() == 1);
        CV_Assert(colormap.channels() == 3);
        CV_Assert(colormap.depth() == CV_8U);

        output_8UC3.create(input_8U.rows, input_8U.cols, CV_8UC3);
        dim3 block(32, 8);
        dim3 grid(cv::cuda::device::divUp(input_8U.cols, block.x),
                  cv::cuda::device::divUp(input_8U.rows, block.y));
        if(input_8U.depth() == CV_8U)
            apply_lut_kernel<uchar><<<grid, block, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(colormap.ptr<cv::Vec3b>(),
                                                                                          input_8U, output_8UC3, colormap.cols);
        else
            apply_lut_kernel<int><<<grid, block, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(colormap.ptr<cv::Vec3b>(),
                                                                                          input_8U, output_8UC3, colormap.cols);
    }
}


