#include "DisplayHelpers.cuh"

#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp> 

template<typename T> 
__global__ void colormap_image(cv::cuda::PtrStepSz<T> image, double alpha, double beta, 
    ColorScale red, ColorScale green, ColorScale blue, cv::cuda::PtrStepSz <uchar3> output)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (x < image.cols && y < image.rows)
    {
        float location = alpha*(float(image(y, x)) - beta);
        uchar3 result;
        result.x = red(location);
        result.y = green(location);
        result.z = blue(location);
        output(y, x) = result;
    }
}


__host__ __device__ ColorScale::ColorScale(double start_, double slope_, bool symmetric_)
{
    start = start_;
    slope = slope_;
    symmetric = symmetric_;
    flipped = false;
    inverted = false;
}
unsigned char __host__ __device__  ColorScale::operator ()(float location)
{
    return getValue(location);
}

unsigned char __host__ __device__  ColorScale::getValue(float location_)
{
    float value = 0;
    if (location_ > start)
    {
        value = (location_ - start)*slope;
    }
    else
    { 
        value = 0;
    }
    if (value > 255)
    {
        if (symmetric) value = 512 - value;
        else value = 255;
    }
    if (value < 0) value = 0;
    if (inverted) value = 255 - value;
    return (unsigned char)value;
}
template<typename T> void dispatcher(cv::cuda::GpuMat& img, double alpha, double beta, ColorScale& red, ColorScale& green, ColorScale& blue, cv::cuda::GpuMat& rgb_out, cv::cuda::Stream& stream)
{
    dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(img.cols, block.x), cv::cuda::device::divUp(img.rows, block.y));
    colormap_image<T> << <grid, block, 0, cv::cuda::StreamAccessor::getStream(stream) >> >(
        cv::cuda::PtrStepSz<T>(img), alpha, beta, red, green, blue, 
        cv::cuda::PtrStepSz<uchar3>(rgb_out));
}


void color_mapper::setMapping(ColorScale red, ColorScale green, ColorScale blue, double min, double max)
{
    red_ = red;
    green_ = green;
    blue_ = blue;
    beta = min;
    alpha = 100/(max - min); 
}

void color_mapper::colormap_image(cv::cuda::GpuMat& img, cv::cuda::GpuMat& rgb_out, cv::cuda::Stream& stream)
{
    CV_Assert(img.channels() == 1);
    rgb_out.create(img.size(), CV_8UC3);
    typedef void(*func_t)(cv::cuda::GpuMat& img, double alpha, double beta, ColorScale& red, ColorScale& green, ColorScale& blue, cv::cuda::GpuMat& rgb_out, cv::cuda::Stream& stream);
    static const func_t funcs[6] = { dispatcher<unsigned char>, dispatcher<char>, dispatcher<unsigned short>, dispatcher<short>, dispatcher<int>, dispatcher<float> };
    funcs[img.depth()](img, alpha, beta, red_, green_, blue_, rgb_out, stream);
}
