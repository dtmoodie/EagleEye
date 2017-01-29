#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev/util/vec_traits.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda/utility.hpp>

#include "EagleLib/utilities/GPUSortingPriv.hpp"

template<class T>
void __global__ accum_kernel(const T* in, float* out)
{
    *out += *in;
}

void __global__ divide_kernel(float* value, float count)
{
    value[0] /= count;
    value[1] /= count;
}

template<class T, class U>
struct Saturate
{
    Saturate(const cv::cuda::PtrStepSz<float>& vals_, float out_dynamic_range):
        vals(vals_), out_dynamic_range(out_dynamic_range)
    {}

    typename cv::cudev::MakeVec<U, 3>::type
    operator()(const typename cv::cudev::MakeVec<T, 3>::type & vec_in)
    {
        typename cv::cudev::MakeVec<U, 3>::type vec;

        float beta = vals(0,0);
        float alpha = out_dynamic_range / (vals(0,1) - vals(0,0));
        float val = alpha * (vec_in.x - beta);
        val = fmax(0.0f, val);
        val = fmin(out_dynamic_range, val);
        vec.x = val;

        beta = vals(1,0);
        alpha = out_dynamic_range / (vals(1,1) - vals(1,0));
        val = alpha * (vec_in.y - beta);
        val = fmax(0.0f, val);
        val = fmin(out_dynamic_range, val);
        vec.y = val;


        beta = vals(2,0);
        alpha = out_dynamic_range / (vals(2,1) - vals(2,0));
        val = alpha * (vec_in.z - beta);
        val = std::max(0.0f, val);
        val = std::min(out_dynamic_range, val);
        vec.z = val;
        return vec;
    }

    const cv::cuda::PtrStepSz<float> vals;
    const float out_dynamic_range;
};

template<class T1, class T2>
void __global__ transform_kernel(const cv::cuda::PtrStepSz<typename cv::cudev::MakeVec<T1, 3>::type> in,
                                cv::cuda::PtrStepSz<typename cv::cudev::MakeVec<T2, 3>::type> out,
                                cv::cuda::PtrStepSz<float> saturate, const float dynamic_range)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x < in.cols && y < in.rows)
    {
        float beta = saturate(0,0);
        float alpha = dynamic_range / (saturate(0,1) - saturate(0,0));
        float val = alpha * (in(y,x).x - beta);
        val = fmax(0.0f, val);
        val = fmin(dynamic_range, val);
        out(y,x).x = val;

        beta = saturate(1,0);
        alpha = dynamic_range / (saturate(1,1) - saturate(0,0));
        val = alpha * (in(y,x).y - beta);
        val = fmax(0.0f, val);
        val = fmin(dynamic_range, val);
        out(y,x).y = val;

        beta = saturate(2,0);
        alpha = dynamic_range / (saturate(2,1) - saturate(2,0));
        val = alpha * (in(y,x).z - beta);
        val = fmax(0.0f, val);
        val = fmin(dynamic_range, val);
        out(y,x).z = val;
    }
}

template<class T1>
void __global__ color_correct_kernel(cv::cuda::PtrStepSz<typename cv::cudev::MakeVec<T1, 3>::type> in,
                                     cv::cuda::PtrStepSz<float> mat)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    typedef typename cv::cudev::MakeVec<T1, 3>::type WorkType;

    if(x < in.cols && y < in.rows)
    {
        WorkType& pix = in(y,x);
        WorkType tmp;
        tmp.x = mat(0,0) * pix.x + mat(0, 1) * pix.y + mat(0,2) * pix.z;
        tmp.y = mat(1,0) * pix.x + mat(1, 1) * pix.y + mat(1,2) * pix.z;
        tmp.z = mat(2,0) * pix.x + mat(2, 1) * pix.y + mat(2,2) * pix.z;
        in(y,x) = tmp;
    }

}

template<class T1, class T2>
void transform(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& out, const cv::cuda::PtrStepSzf& saturate, float dyn_range, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(in.cols, block.x),
              cv::cuda::device::divUp(in.rows, block.y));
    transform_kernel<T1,T2><<<grid, block, 0, stream>>>(in, out, saturate, dyn_range);
}

namespace EagleLib
{
    void applyWhiteBalance(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                           const cv::Scalar& lower, const cv::Scalar& upper,
                           const std::vector<cv::Rect2f>& sample_regions,
                           const std::vector<float>& sample_weights,
                           int dtype, cv::cuda::Stream& stream_)
    {
        CV_Assert(input.channels() == 3);
        CV_Assert(input.depth() == CV_8U || input.depth() == CV_16U);
        for(int i = 0; i < 3; ++i)
        {
            CV_Assert(lower[i] >= 0 && lower[i] < 1.0f);
            CV_Assert(upper[i] >= 0 && upper[i] < 1.0f);
        }
        CV_Assert(sample_regions.size() == sample_weights.size());
        std::vector<float> weights = sample_weights;
        float sum = 0;
        for(int i = 0; i < sample_weights.size(); ++i)
        {
            sum += sample_weights[i];
        }
        sum /= sample_weights.size();
        for(int i = 0; i < weights.size(); ++i)
        {
            weights[i] /= sum;
        }
        int width = input.cols;
        int height = input.rows;
        cudaStream_t stream = cv::cuda::StreamAccessor::getStream(stream_);
        cv::cuda::GpuMat accumulation;
        cv::cuda::createContinuous(3,2, CV_32F, accumulation);
        accumulation.setTo(0.0, stream_);
        float* accum_ptr = accumulation.ptr<float>();

        for(int i = 0; i < sample_regions.size(); ++i)
        {
            const cv::Rect2f& roif = sample_regions[i];
            cv::Rect roi(roif.x*width, roif.y * height, roif.width * width, roif.height * height);
            cv::cuda::GpuMat flat;
            cv::cuda::createContinuous(roi.size(), input.type(), flat);
            input(roi).copyTo(flat, stream_);
            flat = flat.reshape(3, 1);


            for( int j = 0; j < 3; ++j)
            {
                int lower_offset = cvFloor(flat.cols * lower.val[j]) * 3 + j;
                int upper_offset = cvFloor(flat.cols * (1.0f - upper.val[j])) * 3 + j;
                if(input.depth() == CV_8U)
                {
                    auto view = CreateView<uchar, 1>(flat, j);
                    thrust::sort(
                        thrust::system::cuda::par(
                            cv::cuda::device::ThrustAllocator::getAllocator()).on(stream),
                        view.begin(), view.end(), thrust::less<uchar>());

                    accum_kernel<uchar><<<1,1,0, stream>>>(
                                flat.data + lower_offset,
                                accum_ptr + 2*j);

                    accum_kernel<uchar><<<1,1,0, stream>>>(
                                flat.data + upper_offset,
                                accum_ptr + 2*j + 1);

                }else if(input.depth() == CV_16U)
                {
                    auto view = CreateView<ushort, 1>(flat, j);
                    thrust::sort(
                        thrust::system::cuda::par(
                            cv::cuda::device::ThrustAllocator::getAllocator()).on(stream),
                        view.begin(), view.end(), thrust::less<ushort>());

                    accum_kernel<ushort><<<1,1,0, stream>>>(
                                reinterpret_cast<ushort*>(flat.data) + lower_offset,
                                accumulation.ptr<float>(j));

                    accum_kernel<ushort><<<1,1,0, stream>>>(
                                reinterpret_cast<ushort*>(flat.data) + upper_offset,
                                accumulation.ptr<float>(j) + 1);
                }
            }

        }

        if(sample_regions.size() != 1)
        {
            for(int i = 0; i < 3; ++i)
            {
                divide_kernel<<<1,1,0, stream>>>(accumulation.ptr<float>(i), sample_regions.size());
            }
        }
        output.create(input.size(), CV_MAKE_TYPE(dtype, 3));
        typedef void(*func_t)(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& out,
                              const cv::cuda::PtrStepSzf& saturate, float dyn_range, cudaStream_t stream);
        static const func_t funcs[3][3] =
        {
            {transform<uchar, uchar>, 0, transform<ushort, uchar>},
            {0, 0, 0},
            {transform<uchar, ushort>, 0, transform<ushort, ushort>}
        };
        CV_Assert(funcs[dtype][input.depth()]);
        funcs[dtype][input.depth()](input, output, accumulation, 255.0, stream);
    }
    void colorCorrect(cv::cuda::GpuMat& in_out,
                      const cv::cuda::GpuMat& color_matrix,
                      cv::cuda::Stream& stream_)
    {
        CV_Assert(color_matrix.rows == 3 && color_matrix.cols == 3);
        CV_Assert(in_out.channels() == 3);
        CV_Assert(in_out.depth() == CV_8U || in_out.depth() == CV_16U || in_out.depth() == CV_32F);
        dim3 block(32, 8);
        dim3 grid(cv::cuda::device::divUp(in_out.cols, block.x),
                  cv::cuda::device::divUp(in_out.rows, block.y));
        cudaStream_t stream = cv::cuda::StreamAccessor::getStream(stream_);
        switch(in_out.depth())
        {
            case CV_8U: return color_correct_kernel<uchar><<<grid, block, 0, stream>>>(in_out, color_matrix);
            case CV_16U: return color_correct_kernel<ushort><<<grid, block, 0, stream>>>(in_out, color_matrix);
            case CV_32F: return color_correct_kernel<float><<<grid, block, 0, stream>>>(in_out, color_matrix);
        default: return;

        }
    }
}



