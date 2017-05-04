#pragma once
#include "CaffeExport.hpp"
#include <opencv2/core/cuda.hpp>
#include <Aquila/types/SyncedMemory.hpp>
namespace caffe
{
    template<class T> class Blob;
}
namespace aq
{
    namespace Caffe
    {
        Caffe_EXPORT float iou(const cv::Rect& r1, const cv::Rect& r2);
        Caffe_EXPORT void argMax(const caffe::Blob<float>* blob, cv::cuda::GpuMat& label, cv::cuda::GpuMat& confidence, cv::cuda::Stream& stream);
        Caffe_EXPORT void argMax(const caffe::Blob<float>* blob, cv::Mat& label, cv::Mat& confidence);
    }
}
