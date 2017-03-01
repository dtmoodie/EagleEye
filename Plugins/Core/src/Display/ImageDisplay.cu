#include <cuda_runtime.h>
#include <opencv2/cudev.hpp>
#include <device_functions.h>
#include <npp.h>

namespace cv
{
namespace cuda
{
void drawHistogram(cv::InputArray histogram,
                   cv::OutputArray draw,
                   cv::InputArray bins = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null());
}
}


template<int N>
__global__ void draw_uchar_kernel(cv::cudev::PtrStepSz<cv::Vec3b> draw, const int* histogram)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;


    int max_bin = 0;
    int min_bin = NPP_MAX_32S;
    // first find the biggest bin
    for(int i = 0; i < 256 * N; ++i)
    {
        max_bin = max(max_bin, histogram[i]);
        min_bin = min(min_bin, histogram[i]);
    }
    for(int i = 0; i < N; ++i)
    {
        int value = histogram[x * N + i];
        float scaled = float(value - min_bin) / float(max_bin - min_bin);
        int start = draw.rows * (1 - scaled);
        while(start < draw.rows)
        {
            draw(start, x).val[i] = 255;
            ++start;
        }
    }
}

template<int N>
void launch_uchar(cv::cuda::GpuMat& draw, const cv::cuda::GpuMat& hist, cv::cuda::Stream& stream)
{
    CV_Assert(draw.depth() == CV_8U);
    CV_Assert(draw.channels() == N);
    CV_Assert(hist.channels() == N);
    CV_Assert(hist.depth() == CV_32S);
    CV_Assert(hist.cols == 256 && hist.rows == 1);

    dim3 block(256);
    dim3 grid(1);
    draw_uchar_kernel<N><<<1, 256, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(draw, (const int*)hist.data);
}


void cv::cuda::drawHistogram(cv::InputArray histogram,
                   cv::OutputArray draw_,
                   cv::InputArray bins,
                   cv::cuda::Stream& stream)
{
    CV_Assert(histogram.depth() == CV_32S);
    cv::cuda::GpuMat& draw = draw_.getGpuMatRef();
    draw.create(100, 256, CV_MAKE_TYPE(CV_8U, histogram.channels()));
    draw.setTo(cv::Scalar::all(0), stream);
    typedef void(*func_t)(cv::cuda::GpuMat& draw, const cv::cuda::GpuMat& hist, cv::cuda::Stream& stream);
    func_t funcs[4][7] =
    {
        {launch_uchar<1>, 0, 0, 0, 0, 0, 0},
        {launch_uchar<2>, 0, 0, 0, 0, 0, 0},
        {launch_uchar<3>, 0, 0, 0, 0, 0, 0},
        {launch_uchar<4>, 0, 0, 0, 0, 0, 0}
    };
    CV_Assert(funcs[histogram.channels() -1][0]);
    funcs[histogram.channels() - 1][0](draw, histogram.getGpuMat(), stream);
}
