#include "helpers.hpp"
#include "caffe/blob.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/cudev.hpp>
#include <opencv2/core/cuda/transform.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

float EagleLib::Caffe::iou(const cv::Rect& r1, const cv::Rect& r2)
{
    float intersection = (r1 & r2).area();
    float rect_union = (r1.area() + r2.area()) - intersection;
    return intersection / rect_union;
}

template<typename T> struct Matrix3D
{
    int channels, height, width;
    T* data;
    __host__ __device__ T& operator()(int c, int h, int w)
    {
        return data[c*height*width + h*width + w];
    }
};

void __global__ argmaxKernel(Matrix3D<const float> data, cv::cuda::PtrStepSz<float> confidence, cv::cuda::PtrStepSz<uchar> label)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    float maxValue = 0;
    uchar maxLabel = 0;
    for(int i = 0; i < data.channels; ++i)
    {
        if(data(i, y, x) > maxValue)
        {
            maxValue = data(i, y, x);
            maxLabel = i;
        }
    }
    confidence(y,x) = maxValue;
    label(y,x) = maxLabel;
}

void EagleLib::Caffe::argMax(const caffe::Blob<float>* blob_, cv::Mat& h_label, cv::Mat& h_confidence)
{
    auto shape = blob_->shape();
    CV_Assert(shape.size() == 4);
    Matrix3D<const float> cpu_blob;
    cpu_blob.data = blob_->cpu_data();
    cpu_blob.channels = shape[1];
    cpu_blob.height = shape[2];
    cpu_blob.width = shape[3];
    CV_Assert(cpu_blob.channels < 255);
    h_label.create(shape[2], shape[3], CV_8U);
    h_confidence.create(shape[2], shape[3], CV_32F);
    for(int i = 0; i < cpu_blob.height; ++i)
    {
        for(int j = 0; j < cpu_blob.width; ++j)
        {
            float val = 0;
            int idx = 0;
            for(int c = 0; c < cpu_blob.channels; ++c)
            {
                if(cpu_blob(c,i, j) > val)
                {
                    val = cpu_blob(c, i, j);
                    idx = c;
                }
            }
            h_label.at<uchar>(i,j) = idx;
            h_confidence.at<float>(i,j) = val;
        }
    }
}

void EagleLib::Caffe::argMax(const caffe::Blob<float>* blob_, cv::cuda::GpuMat& label, cv::cuda::GpuMat& confidence, cv::cuda::Stream& stream_)
{
    const float* data = blob_->gpu_data();
    auto shape = blob_->shape();
    CV_Assert(shape.size() == 4);
    Matrix3D<const float> blob;
    blob.data = data;
    blob.channels = shape[1];
    blob.height = shape[2];
    blob.width = shape[3];
    label.create(shape[2], shape[3], CV_8U);
    confidence.create(shape[2], shape[3], CV_32F);
    dim3 threads(16, 16, 1);

    dim3 blocks(cv::cudev::divUp(shape[3], 16),
                cv::cudev::divUp(shape[2], 16),
                1);
    auto stream = cv::cuda::StreamAccessor::getStream(stream_);

    argmaxKernel<<<blocks, threads, 0, stream>>>(blob, confidence, label);
}
