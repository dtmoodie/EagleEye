

#include <cuda_runtime_api.h>
#include <opencv2/cudev.hpp>
#include <opencv2/core/matx.hpp>
#include <EagleLib/Thrust_interop.hpp>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>
namespace cv
{
namespace cuda
{
void histogram(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& bins, cv::cuda::GpuMat& histogram,
               float min = 0, float max = 256,
               cv::cuda::Stream& stream = cv::cuda::Stream::Null());
}
}
template<typename T>
__host__ __device__ inline T* binary_search_approx(T *const begin, T * end, T value)
{
    T* q;
    if(begin >= end)
    {
        return end;
    }
    //q = (begin + end) / 2;
    q = begin + (end - begin) / 2;
    if(value == *q)
    {
        return q;
    }else if(value > *q)
    {
        return binary_search_approx(q + 1, end, value);
    }else if(value < *q)
    {
        return binary_search_approx(begin, q - 1, value);
    }
}


template<typename T, int N>
__global__ void histogram_kernel(const cv::cuda::PtrStepSz<cv::Vec<T, N>> input,
                                 const cv::cuda::PtrStepSz<float> bins,
                                 cv::cuda::PtrStepSz<cv::Vec<int, N>> histogram)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int t = threadIdx.x + threadIdx.y * blockDim.x;

    int nt = blockDim.x * blockDim.y;

    const int num_bins = bins.cols;

    // linear block index within 2D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    //extern __shared__ int smem[N * num_bins + N];
    extern __shared__ int smem[];

    for (int i = t; i < N * num_bins + N; i += nt)
        smem[i] = 0;

    __syncthreads();

    for (int row = y; row < input.rows; row += ny)
    {
        for (int col = x; col < input.cols; col += nx)
        {
#pragma unroll
            for(int c = 0; c < N; ++c)
            {
                T val = input(row, col).val[c];
                float* bin = binary_search_approx<float>(bins.data, bins.data + bins.cols, float(val));
                int dist = bin - bins.data;
                atomicAdd(&smem[dist * N + c], 1);
            }
        }
    }
      __syncthreads();


    for (int i = t; i < num_bins; i += nt) {
#pragma unroll
        for(int c = 0; c < N; ++c)
        {
            histogram(0,i).val[c] = smem[i * N + c];
        }
    }
}

template<int N>
__global__ void histogram_kernel_uchar(const cv::cuda::PtrStepSz<cv::Vec<uchar, N>> input,
                                 int* histogram)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int t = threadIdx.x + threadIdx.y * blockDim.x;

    int nt = blockDim.x * blockDim.y;

    const int num_bins = 256;

    // linear block index within 2D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    //extern __shared__ int smem[N * num_bins + N];
    extern __shared__ int smem[];

    for (int i = t; i < N * num_bins + N; i += nt)
        smem[i] = 0;

    __syncthreads();

    for (int row = y; row < input.rows; row += ny)
    {
        for (int col = x; col < input.cols; col += nx)
        {
#pragma unroll
            for(int c = 0; c < N; ++c)
            {
                uchar val = input(row, col).val[c];
                atomicAdd(&smem[val* N + c], 1);
            }
        }
    }
      __syncthreads();


    for (int i = t; i < num_bins; i += nt) {
#pragma unroll
        for(int c = 0; c < N; ++c)
        {
            atomicAdd(histogram + i * N + c, smem[i * N + c]);
        }
    }
}

template<class T, int N>
void launch(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& bins, cv::cuda::GpuMat& hist, cv::cuda::Stream& stream)
{
    dim3 block(16, 16);
    dim3 grid(cv::cudev::divUp(in.cols,16), cv::cudev::divUp(in.rows, 16));
    histogram_kernel<T,N><<<grid, block, bins.cols * N + N,
            cv::cuda::StreamAccessor::getStream(stream)>>>(
                in, bins, hist);
}
template<int N>
void launch_uchar(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& bins, cv::cuda::GpuMat& hist, cv::cuda::Stream& stream)
{
    CV_Assert(in.depth() == CV_8U);
    CV_Assert(in.channels() == N);
    CV_Assert(hist.cols == 256 && hist.rows == 1 && hist.depth() == CV_32S && hist.channels() == N);
    dim3 block(16, 16);
    dim3 grid(cv::cudev::divUp(in.cols,16), cv::cudev::divUp(in.rows, 16));
    histogram_kernel_uchar<N><<<grid, block, (256 * N + N) * sizeof(int),
            cv::cuda::StreamAccessor::getStream(stream)>>>(
                in, (int*)hist.data);
}

void cv::cuda::histogram(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& bins, cv::cuda::GpuMat& histogram,
                         float min, float max,
                         cv::cuda::Stream& stream)
{
    typedef void(*func_t)(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& bins, cv::cuda::GpuMat& hist, cv::cuda::Stream& stream);
    int size = 1000;
    if(in.depth() == CV_8U)
    {
        size = 256;
        min = 0;
        max = 256;
    }
    if(bins.empty() && in.depth() != CV_8U)
    {
        bins.create(1, size, CV_32F);
        float step = (max - min) / float(size);
        thrust::device_ptr<float> ptr = thrust::device_pointer_cast((float*)bins.data);
        thrust::sequence(thrust::system::cuda::par.on(cv::cuda::StreamAccessor::getStream(stream)),ptr, ptr + size, min, step);
    }
    histogram.create(1, size, CV_MAKE_TYPE(CV_32S, in.channels()));
    histogram.setTo(cv::Scalar::all(0), stream);
    func_t funcs[4][7] =
    {
        {launch_uchar<1>, 0, launch<ushort, 1>, 0, 0, 0, 0},
        {launch_uchar<2>, 0, launch<ushort, 2>, 0, 0, 0, 0},
        {launch_uchar<3>, 0, launch<ushort, 3>, 0, 0, 0, 0},
        {launch_uchar<4>, 0, launch<ushort, 4>, 0, 0, 0, 0}
    };
    CV_Assert(funcs[in.channels() - 1][in.depth()]);

    funcs[in.channels() - 1][in.depth()](in, bins, histogram, stream);
}
