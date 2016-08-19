#pragma once

#include "EagleLib/Detail/Export.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
namespace EagleLib
{
    class EAGLE_EXPORTS IColorMapper
    {
    public:
        virtual ~IColorMapper();

        // Apply a colormap to an input cpu or gpu image, with the output being a passed in buffer
        virtual void Apply(cv::InputArray input, cv::OutputArray output, cv::InputArray mask = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null()) = 0;

        // Apply a colormap to an input cpu or gpu image with the output being a returned gpu mat
        virtual cv::cuda::GpuMat Apply(cv::cuda::GpuMat input, cv::InputArray mask = cv::noArray(), cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual cv::Mat Apply(cv::Mat input, cv::InputArray mask = cv::noArray());
        virtual void Rescale(float alpha, float beta) = 0;
        // Returns a matrix where
        // Column(0) = x location
        // Column(1) = r location
        // Column(2) = g location
        // Column(3) = b location
        // input min is the min x location
        // input max is the max x location
        // resolution is the number of samples to estimate
        virtual cv::Mat_<float> GetMat(float min, float max, int resolution) = 0;
    };
}