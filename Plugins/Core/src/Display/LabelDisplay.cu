
#include <EagleLib/Thrust_interop.hpp>
#include <vector_types.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda/utility.hpp>
#include <opencv2/core/cuda/type_traits.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
struct ApplyLut
{
    ApplyLut(const cv::Vec3b* lut_):
        lut(lut_)
    {

    }
    template<class T>
    __device__ __host__ void operator()(T zip)
    {
        thrust::get<1>(zip) = lut[thrust::get<0>(zip)];
    }
    __device__ __host__ cv::Vec3b operator()(const uchar& val)
    {
        return lut[val];
    }

    const cv::Vec3b* lut;
};

void __global__ apply_lut_kernel(const cv::Vec3b* lut, const cv::cuda::PtrStepSzb in, cv::cuda::PtrStepSz<cv::Vec3b> out)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < in.cols && y < in.rows)
    {
        out(y,x) = lut[in(y,x)];
    }
}

namespace EagleLib
{
    void applyColormap(const cv::cuda::GpuMat& input_8U,
                                 cv::cuda::GpuMat& output_8UC3,
                                 const cv::cuda::GpuMat& colormap,
                                 cv::cuda::Stream& stream)
    {
        CV_Assert(input_8U.size().area());
        CV_Assert(input_8U.depth() == CV_8U);
        CV_Assert(input_8U.channels() == 1);
        CV_Assert(colormap.channels() == 3);
        CV_Assert(colormap.depth() == CV_8U);

        output_8UC3.create(input_8U.rows, input_8U.cols, CV_8UC3);
        dim3 block(32, 8);
        dim3 grid(cv::cuda::device::divUp(input_8U.cols, block.x),
                  cv::cuda::device::divUp(input_8U.rows, block.y));
        apply_lut_kernel<<<grid, block, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(colormap.ptr<cv::Vec3b>(),
                                                                                          input_8U, output_8UC3);
        /*ThrustView<const uchar, 1, thrust::device_ptr<const uchar>> input = CreateView<uchar, 1>(input_8U, -1);
        ThrustView<cv::Vec3b, 1, thrust::device_ptr<cv::Vec3b>> output = CreateView<cv::Vec3b, 1>(output_8UC3, -1);*/

        /*auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(input.begin(), output.begin()));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(input.end(), output.end()));
        thrust::for_each(thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(cv::cuda::StreamAccessor::getStream(stream)),
                         zip_begin, zip_end, ApplyLut(colormap.ptr<cv::Vec3b>()));*/

        /*thrust::transform(thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(cv::cuda::StreamAccessor::getStream(stream)),
                          input.begin(), input.end(), output.begin(), ApplyLut(colormap.ptr<cv::Vec3b>()));*/

    }
}


