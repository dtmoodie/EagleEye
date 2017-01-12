#pragma once
#include "EagleLib/utilities/GPUSorting.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/utility.hpp>
#include <cuda_runtime_api.h>
#include <EagleLib/Thrust_interop.hpp>

#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

namespace cv
{
    namespace cuda
    {
        namespace detail
        {
            template<typename T> struct UnarySortDescending
            {
                template<typename U1> void operator()(const U1& it1)
                {
                    thrust::sort(&thrust::get<0>(it1), &thrust::get<1>(it1), thrust::greater<T>());
                }
            };

            template<typename T> struct UnarySortAscending
            {
                template<typename U1> void operator()(const U1& it1)
                {
                    thrust::sort(&thrust::get<0>(it1), &thrust::get<1>(it1), thrust::less<T>());
                }
            };

            template<typename T> void sortAscending(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
                thrust::sort(thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream), 
                    view.begin(), view.end(), thrust::less<float>());
            }

            template<typename T> void sortDescending(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
                thrust::sort(thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream), 
                    view.begin(), view.end(), thrust::greater<float>());
            }

            template<typename T> void sortAscendingEachCol(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T>(out);

            }

            template<typename T> void sortDescendingEachCol(cv::cuda::GpuMat& out, cudaStream_t stream)
            {

            }

            template<typename T> void sortAscendingEachRow(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
                auto range = view.rowRange(0, -1);
                thrust::for_each(thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream), 
                    range.first, range.second, UnarySortAscending<T>());
            }

            template<typename T> void sortDescendingEachRow(cv::cuda::GpuMat& out, cudaStream_t stream)
            {
                auto view = CreateView<T, 1>(out);
                auto range = view.rowRange(0, -1);
                thrust::for_each(thrust::system::cuda::par(cv::cuda::device::ThrustAllocator::getAllocator()).on(stream), 
                    range.first, range.second, UnarySortDescending<T>());
            }
        }
    } // namespace cuda
} // namespace cv
