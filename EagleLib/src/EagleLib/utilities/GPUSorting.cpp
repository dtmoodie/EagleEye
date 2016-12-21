#include <EagleLib/utilities/GPUSorting.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace cv::cuda::detail;
void cv::cuda::sort(InputArray src, OutputArray dst, int flags, cv::cuda::Stream& stream)
{
    const cv::cuda::GpuMat src_ = src.getGpuMat();
    cv::cuda::GpuMat& dst_ = dst.getGpuMatRef();
    // Check if the sorting should be done in place
    if(dst_.data != src_.data)
    {
        src_.copyTo(dst_, stream);
    }
    typedef void(*func_t)(cv::cuda::GpuMat&, cudaStream_t);
    func_t ascendSorts[] =
    {
        sortAscending<uchar>, sortAscending<char>, sortAscending<ushort>,
        sortAscending<short>, sortAscending<int>, sortAscending<float>, sortAscending<double>
    };
    func_t descendSorts[] =
    {
        sortDescending<uchar>, sortDescending<char>, sortDescending<ushort>,
        sortDescending<short>, sortDescending<int>, sortDescending<float>, sortDescending<double>
    };
    func_t ascendEachRowSorts[] =
    {
        sortAscendingEachRow<uchar>, sortAscendingEachRow<char>, sortAscendingEachRow<ushort>,
        sortAscendingEachRow<short>, sortAscendingEachRow<int>, sortAscendingEachRow<float>, sortAscendingEachRow<double>
    };
    func_t descendEachRowSorts[] =
    {
        sortDescendingEachRow<uchar>, sortDescendingEachRow<char>, sortDescendingEachRow<ushort>,
        sortDescendingEachRow<short>, sortDescendingEachRow<int>, sortDescendingEachRow<float>, sortDescendingEachRow<double>
    };
    if (flags & cv::SORT_DESCENDING)
    {
        if(flags & cv::SORT_EVERY_ROW)
        {
            descendEachRowSorts[dst.depth()](dst_, cv::cuda::StreamAccessor::getStream(stream));
        }else if(flags & cv::SORT_EVERY_COLUMN)
        {
            cv::error(1, "Column sorting not yet implemented on the GPU", __FUNCTION__, __FILE__, __LINE__);
        }
        else
        {
            descendSorts[dst.depth()](dst_, cv::cuda::StreamAccessor::getStream(stream));
        }
    }else 
    {
        if (flags & cv::SORT_EVERY_ROW)
        {
            ascendEachRowSorts[dst.depth()](dst_, cv::cuda::StreamAccessor::getStream(stream));
        }
        else if (flags & cv::SORT_EVERY_COLUMN)
        {
            cv::error(1, "Column sorting not yet implemented on the GPU", __FUNCTION__, __FILE__, __LINE__);
        }
        else
        {
            ascendSorts[dst.depth()](dst_, cv::cuda::StreamAccessor::getStream(stream));
        }
    }

}

