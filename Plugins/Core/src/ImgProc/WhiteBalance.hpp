#pragma once
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/SyncedMemory.hpp>
#include <opencv2/imgproc.hpp>
namespace aq
{
    //void ApplyWhiteBalance(const cv::cuda::GpuMat& in_8uc3, cv::cuda::GpuMat& out_8uc3, )
    void applyWhiteBalance(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                       const cv::Scalar& lower, const cv::Scalar& upper,
                       const std::vector<cv::Rect2f>& sample_regions,
                       const std::vector<float>& sample_weights,
                       int dtype, cv::cuda::Stream& stream);
    // Inplace color correction
    void colorCorrect(cv::cuda::GpuMat& input_output,
                      const cv::cuda::GpuMat& color_matrix,
                      cv::cuda::Stream& stream);
    namespace Nodes
    {
        class WhiteBalance: public Node
        {
        public:


            MO_DERIVE(WhiteBalance, Node)
                INPUT(SyncedMemory, input, nullptr)
                //PARAM(cv::Scalar, upper_percent, cv::Scalar::all(0.2))
                //PARAM(cv::Scalar, lower_percent, cv::Scalar::all(0.1))
                PARAM(float, upper_red, 0.05)
                PARAM(float, upper_green, 0.05)
                PARAM(float, upper_blue, 0.05)
                PARAM(float, lower_red, 0.05)
                PARAM(float, lower_green, 0.05)
                PARAM(float, lower_blue, 0.05)
                PARAM(int, dtype, -1)
                PARAM(std::vector<cv::Rect2f>, rois, {})
                PARAM(std::vector<float>, weight, {})

                OUTPUT(SyncedMemory, output, {})
            MO_END
            protected:
                bool processImpl();
        };
        class StaticWhiteBalance: public Node
        {
        public:
            MO_DERIVE(StaticWhiteBalance, Node)
                INPUT(SyncedMemory, input, nullptr)
                OUTPUT(SyncedMemory, output, {})
                PARAM(cv::Scalar, low, cv::Scalar(27,27,27))
                PARAM(cv::Scalar, high, cv::Scalar(160, 180, 160))
                PARAM(int, dtype, -1)
                PARAM(float, min, 0)
                PARAM(float, max, 255)
            MO_END
            protected:
                bool processImpl();
        };
        class WhiteBalanceMean: public Node
        {
        public:
            MO_DERIVE(WhiteBalanceMean, Node)
                INPUT(SyncedMemory, input, nullptr)
                OUTPUT(SyncedMemory, output, {})
                PARAM(float, K, 0.8)
            MO_END
        protected:
            bool processImpl();
            cv::Mat h_m;
            cv::cuda::GpuMat d_m;
        };
    }
}
