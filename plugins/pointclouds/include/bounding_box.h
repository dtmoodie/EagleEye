#pragma once

#include "pc_export.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace aq
{
    namespace point_clouds
    {
        
        class PC_EXPORTS bounding_box
        {
        public:
            bounding_box(cv::Mat transform, cv::Scalar size);
            bool contains(const cv::Vec3f& point);
            cv::Mat contains(const cv::Mat& points);
            cv::cuda::GpuMat contains(const cv::cuda::GpuMat& points, cv::cuda::Stream& stream);
            void contains(cv::InputArray points_in, cv::InputOutputArray mask_out, cv::cuda::Stream& stream);
        protected:
            cv::Mat _transform;
            cv::Mat _inv_transform;
            cv::Scalar _size;
        };
    }
}