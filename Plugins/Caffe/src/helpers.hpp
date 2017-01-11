#pragma once
#include <opencv2/core/cuda.hpp>
#include <EagleLib/SyncedMemory.h>
namespace caffe
{
    template<class T> class Blob;
}
namespace EagleLib
{
    namespace Caffe
    {
        float iou(const cv::Rect& r1, const cv::Rect& r2);
        void MaxSegmentation(const caffe::Blob<float>* blob, cv::cuda::GpuMat& label, cv::cuda::GpuMat& confidence, cv::cuda::Stream& stream);
        void MaxSegmentation(const caffe::Blob<float>* blob, cv::Mat& label, cv::Mat& confidence);
    }
}
