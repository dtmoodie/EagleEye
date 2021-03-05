#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev.hpp>

namespace cv
{
    namespace cuda
    {
        void drawHistogram(cv::InputArray histogram,
                           cv::OutputArray draw,
                           cv::InputArray bins = cv::noArray(),
                           cv::cuda::Stream& stream = cv::cuda::Stream::Null());

        void
        drawPlot(cv::InputArray histogram, cv::OutputArray draw, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    } // namespace cuda
} // namespace cv

template <int N>
__global__ void draw_uchar_kernel(cv::cudev::PtrStepSz<cv::Vec3b> draw, const int* histogram, int size)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int max_bin = 0;
    int min_bin = NPP_MAX_32S;
    // first find the biggest bin
    for (int i = 0; i < size * N; ++i)
    {
        max_bin = max(max_bin, histogram[i]);
        min_bin = min(min_bin, histogram[i]);
    }
    for (int i = 0; i < N; ++i)
    {
        int value = histogram[x * N + i];
        float scaled = float(value - min_bin) / float(max_bin - min_bin);
        int start = draw.rows * (1 - scaled);
        while (start < draw.rows)
        {
            draw(start, x).val[i] = 255;
            ++start;
        }
    }
}

template <class T, int N>
__global__ void draw_kernel(cv::cudev::PtrStepSz<cv::Vec3b> draw, const T* histogram, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    T max_bin = histogram[0];
    T min_bin = histogram[0];
    // first find the biggest bin
    for (int i = 1; i < size * N; ++i)
    {
        max_bin = max(max_bin, histogram[i]);
        min_bin = min(min_bin, histogram[i]);
    }
    for (int i = 0; i < N; ++i)
    {
        float value = histogram[x * N + i];
        float scaled = float(value - min_bin) / float(max_bin - min_bin);
        int start = draw.rows * (1 - scaled);
        start = min(draw.rows - 1, max(0, start));
        while (start < draw.rows)
        {
            draw(start, x).val[i] = 255;
            ++start;
        }
    }
}

template <class T>
__global__ void draw_kernel(cv::cudev::PtrStepSz<cv::Vec3b> draw, const T* histogram, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
    {
        T max_bin = histogram[0];
        T min_bin = histogram[0];
        // first find the biggest bin
        for (int i = 1; i < size; ++i)
        {
            max_bin = max(max_bin, histogram[i]);
            min_bin = min(min_bin, histogram[i]);
        }

        T value = histogram[x];
        float scaled = float(value - min_bin) / float(max_bin - min_bin);
        int start = draw.rows * (1 - scaled);
        start = min(draw.rows - 1, max(0, start));
        while (start < draw.rows)
        {
#pragma unroll
            for (int i = 0; i < 3; ++i)
                draw(start, x).val[i] = 255;
            ++start;
        }
    }
}

template <int N>
void launch_uchar(cv::cuda::GpuMat& draw, const cv::cuda::GpuMat& hist, cv::cuda::Stream& stream)
{
    CV_Assert(draw.depth() == CV_8U);
    CV_Assert(draw.channels() == N);
    CV_Assert(hist.channels() == N);
    CV_Assert(hist.depth() == CV_32S);
    CV_Assert(hist.rows == 1);

    draw_uchar_kernel<N>
        <<<1, 256, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(draw, (const int*)hist.data, hist.cols);
}

template <class T, int N>
void launch(cv::cuda::GpuMat& draw, const cv::cuda::GpuMat& hist, cv::cuda::Stream& stream)
{
    CV_Assert(draw.depth() == CV_8U);
    CV_Assert(draw.channels() == 3);
    CV_Assert(hist.channels() == N);
    CV_Assert(hist.depth() == cv::DataType<T>::depth);
    CV_Assert(hist.rows == 1);
    cudaStream_t _stream = cv::cuda::StreamAccessor::getStream(stream);
    draw_kernel<T, N><<<1, 256, 0, _stream>>>(draw, (const T*)hist.data, hist.cols);
}

template <class T>
void launch(cv::cuda::GpuMat& draw, const cv::cuda::GpuMat& hist, cv::cuda::Stream& stream)
{
    CV_Assert(draw.depth() == CV_8U);
    CV_Assert(draw.channels() == 3);
    CV_Assert(hist.channels() == 1);
    CV_Assert(hist.depth() == cv::DataType<T>::depth);
    CV_Assert(hist.rows == 1);
    cudaStream_t _stream = cv::cuda::StreamAccessor::getStream(stream);
    draw_kernel<T><<<1, 256, 0, _stream>>>(draw, (const T*)hist.data, hist.cols);
}
void cv::cuda::drawHistogram(cv::InputArray histogram,
                             cv::OutputArray draw_,
                             cv::InputArray bins,
                             cv::cuda::Stream& stream)
{
    drawPlot(histogram, draw_, stream);
}

void cv::cuda::drawPlot(cv::InputArray arr, cv::OutputArray draw_, cv::cuda::Stream& stream)
{
    CV_Assert(arr.depth() == CV_32S || arr.depth() == CV_32F);
    cv::cuda::GpuMat& draw = draw_.getGpuMatRef();
    if (draw.rows != 100 || draw.cols != arr.cols())
    {
        draw.create(100, arr.cols(), CV_MAKE_TYPE(CV_8U, arr.channels()));
    }

    draw.setTo(cv::Scalar::all(0), stream);
    typedef void (*func_t)(cv::cuda::GpuMat & draw, const cv::cuda::GpuMat& hist, cv::cuda::Stream& stream);
    func_t funcs[4][7] = {// CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F
                          {launch<uchar>, 0, 0, 0, launch<int>, launch<float>, 0},
                          {launch<uchar, 2>, 0, 0, 0, launch<int, 2>, launch<float, 2>, 0},
                          {launch<uchar, 3>, 0, 0, 0, launch<int, 3>, launch<float, 3>, 0},
                          {launch<uchar, 4>, 0, 0, 0, launch<int, 4>, launch<float, 4>, 0}};
    CV_Assert(funcs[arr.channels() - 1][arr.depth()]);
    funcs[arr.channels() - 1][arr.depth()](draw, arr.getGpuMat(), stream);
}
