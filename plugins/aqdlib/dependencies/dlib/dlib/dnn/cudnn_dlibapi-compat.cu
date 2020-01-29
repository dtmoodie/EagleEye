#include <cudnn.h>
#include "tensor.h"

namespace dlib
{
namespace device
{
    __global__ void padded_dot_product_kernel(float* dest,
                                         const float* src,
                                         const unsigned int batch_size,
                                         const unsigned int src_channels,
                                         const float* filter_weight,
                                         const float beta,
                                         const unsigned int filter_weight_stride,
                                         const unsigned int filter_channels)
    {
        unsigned int filter_output_channel = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int batch = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned int filter_weight_offset = filter_output_channel * filter_channels * filter_weight_stride;

        float activation = 0.f;
        for(unsigned int i = 0; i < src_channels; ++i)
        {
            activation += src[batch * src_channels + i] * filter_weight[i * filter_weight_stride + filter_weight_offset];
        }
        dest[batch * filter_channels + filter_output_channel] = activation + beta;

    }

    // ______
    // |0 1 2|
    // |3 4 5|
    // |6 7 8|
    void padded_dot_convolve(tensor& dest,
                                   const tensor& src,
                                   const tensor& filter,
                                    const float beta)
    {
        DLIB_CASSERT(dest.nr() == 1);
        DLIB_CASSERT(dest.nc() == 1);
        DLIB_CASSERT(src.nr() == 1);
        DLIB_CASSERT(src.nc() == 1);
        DLIB_CASSERT(dest.num_samples() == src.num_samples());
        DLIB_CASSERT(filter.nc() == 3);
        DLIB_CASSERT(filter.nr() == 3);
        DLIB_CASSERT(filter.num_samples() == dest.k());
        DLIB_CASSERT(filter.k() == src.k());
        dim3 blocks;
        blocks.x = filter.num_samples();
        blocks.y = src.num_samples();
        blocks.z = 1;
        const float* filter_ptr = filter.device();
        filter_ptr += 4;
        padded_dot_product_kernel<<<blocks, 1, 0>>>(dest.device(), src.device(), dest.num_samples(), src.k(), filter_ptr, beta, 9, filter.num_samples());

    }
}
}
