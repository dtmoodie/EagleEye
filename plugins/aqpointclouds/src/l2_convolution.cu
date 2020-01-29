#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudev/common.hpp>

__device__ float norm(float3 pt1, float3 pt2)
{
    float x = pt1.x - pt2.x;
    float y = pt1.y - pt2.y;
    float z = pt1.z - pt2.z;
    return x * x + y * y + z * z;
}


template <int K>
__global__ void static_size_kernel(cv::cuda::PtrStepSz<float3> input,
                        cv::cuda::PtrStepSz<float> distance,
                        cv::cuda::PtrStepSz<int> index,
                        float distance_threshold)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float min_dist = 10000000.0f;
    int min_idx = -1;
    float3 pt0 = input(y,x);
    for(unsigned int j = y - K / 2; j < y + K / 2; ++j)
    {
        if(j > 0 && j < input.rows)
        {
            for(unsigned int i = x - K / 2; i < x + K / 2; ++i)
            {
                if(i > 0 && i < input.cols && x != i && y != j)
                {
                    float3 pt1 = input(j,i);
                    float dist = norm(pt0, pt1);
                    if(dist < min_dist)
                    {
                        min_dist = dist;
                        min_idx = j * input.cols + i;
                    }
                }
            }
        }
    }
    if(min_dist > distance_threshold)
    {
        distance(y,x) = min_dist;
        index(y,x) = min_idx;
    }else
    {
        distance(y,x) = 0;
        index(y,x) = -1;
    }
}

template <int K>
__global__ void static_size_kernel(cv::cuda::PtrStepSz<float3> input1,
                                   cv::cuda::PtrStepSz<float3> input2,
                        cv::cuda::PtrStepSz<float> distance,
                        cv::cuda::PtrStepSz<int> index,
                        float distance_threshold)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float min_dist = 10000000.0f;
    int min_idx = -1;
    float3 pt0 = input1(y,x);
    for(unsigned int j = y - K / 2; j < y + K / 2; ++j)
    {
        if(j > 0 && j < input1.rows)
        {
            for(unsigned int i = x - K / 2; i < x + K / 2; ++i)
            {
                if(i > 0 && i < input1.cols)
                {
                    float3 pt1 = input2(j,i);
                    float dist = norm(pt0, pt1);
                    if(dist < min_dist)
                    {
                        min_dist = dist;
                        min_idx = j * input1.cols + i;
                    }
                }
            }
        }
    }
    if(min_dist > distance_threshold)
    {
        distance(y,x) = min_dist;
        index(y,x) = min_idx;
    }else
    {
        distance(y,x) = 0;
        index(y,x) = -1;
    }
}

template <int K>
void launch_static_sized(cv::cuda::GpuMat& output,
                         cv::cuda::GpuMat& index,
                         const cv::cuda::GpuMat& input,
                         float dist,
                         cv::cuda::Stream& stream)
{
    dim3 block(16, 16);
    dim3 grid(cv::cudev::divUp(input.cols, 16), cv::cudev::divUp(input.rows, 16));
    static_size_kernel<K><<<grid, block, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(
        input, output, index, dist);
}

template <int K>
void launch_static_sized(cv::cuda::GpuMat& output,
                         cv::cuda::GpuMat& index,
                         const cv::cuda::GpuMat& input1,
                         const cv::cuda::GpuMat& input2,
                         float dist,
                         cv::cuda::Stream& stream)
{
    dim3 block(16, 16);
    dim3 grid(cv::cudev::divUp(input1.cols, 16), cv::cudev::divUp(input1.rows, 16));
    static_size_kernel<K><<<grid, block, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(
        input1, input2, output, index, dist);
}

namespace aq
{
    namespace pointclouds
    {
        namespace device
        {

            void convolveL2(cv::cuda::GpuMat& output_distance,
                            cv::cuda::GpuMat& output_index,
                            const cv::cuda::GpuMat& input,
                            int ksize,
                            float distance_threshold,
                            cv::cuda::Stream& stream)
            {
                CV_Assert(input.depth() == CV_32F);
                CV_Assert(input.channels() == 3);
                output_distance.create(input.size(), CV_32F);
                output_index.create(input.size(), CV_32F);
                typedef void (*func)(cv::cuda::GpuMat & output,
                                      cv::cuda::GpuMat & index,
                                      const cv::cuda::GpuMat& input,
                                      float dist,
                                      cv::cuda::Stream& stream);
                if(ksize % 2 != 1)
                {
                    ksize += 1;
                }
                if(ksize == 3)
                {
                    ksize = 0;
                }else
                {
                    ksize = ksize / 3;
                }

                func funcs[4] = {&launch_static_sized<3>,
                                 &launch_static_sized<5>,
                                 &launch_static_sized<7>,
                                 &launch_static_sized<9>};
                funcs[ksize](output_distance, output_index, input, distance_threshold, stream);
            }

            void convolveL2(cv::cuda::GpuMat& output_distance,
                            cv::cuda::GpuMat& output_index,
                            const cv::cuda::GpuMat& input1,
                            const cv::cuda::GpuMat& input2,
                            int ksize,
                            float distance_threshold,
                            cv::cuda::Stream& stream)
            {
                CV_Assert(input1.depth() == CV_32F);
                CV_Assert(input1.channels() == 3);
                CV_Assert(input1.type() == input2.type());
                CV_Assert(input1.size() == input2.size());

                output_distance.create(input1.size(), CV_32F);
                output_index.create(input1.size(), CV_32F);
                typedef void (*func)(cv::cuda::GpuMat & output,
                                      cv::cuda::GpuMat & index,
                                      const cv::cuda::GpuMat& input1,
                                     const cv::cuda::GpuMat& input2,
                                      float dist,
                                      cv::cuda::Stream& stream);
                if(ksize % 2 != 1)
                {
                    ksize += 1;
                }
                if(ksize == 3)
                {
                    ksize = 0;
                }else
                {
                    ksize = ksize / 3;
                }

                func funcs[4] = {&launch_static_sized<3>,
                                 &launch_static_sized<5>,
                                 &launch_static_sized<7>,
                                 &launch_static_sized<9>};
                funcs[ksize](output_distance, output_index, input1, input2, distance_threshold, stream);
            }
        }
    }
}
