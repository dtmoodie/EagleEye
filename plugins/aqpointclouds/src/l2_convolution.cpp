#include <Aquila/nodes/NodeInfo.hpp>
#include <l2_convolution.hpp>

namespace aq
{
    namespace pointclouds
    {
        namespace device
        {
            void convolveL2(cv::cuda::GpuMat& output_distance,
                            cv::cuda::GpuMat& output_index,
                            const cv::cuda::GpuMat& input,
                            int ksize,
                            float distance_threshold,
                            cv::cuda::Stream& stream);

            void convolveL2(cv::cuda::GpuMat& output_distance,
                            cv::cuda::GpuMat& output_index,
                            const cv::cuda::GpuMat& in1,
                            const cv::cuda::GpuMat& in2,
                            int ksize,
                            float distance_threshold,
                            cv::cuda::Stream& stream);
        }

        bool ConvolutionL2::processImpl()
        {
            cv::cuda::GpuMat dist, idx;
            device::convolveL2(dist, idx, input->getGpuMat(stream()), kernel_size, distance_threshold, stream());
            distance_param.updateData(dist, mo::tag::_param = input_param, mo::tag::_context = _ctx.get());
            index_param.updateData(idx, mo::tag::_param = input_param, mo::tag::_context = _ctx.get());
            return true;
        }

        void ConvolutionL2ForegroundEstimate::buildModel() { this->build_model = true; }

        bool ConvolutionL2ForegroundEstimate::processImpl()
        {
            if (build_model)
            {
                prev = input->getGpuMat(stream());
                build_model = false;
            }
            if (!prev.empty())
            {
                cv::cuda::GpuMat dist, idx;
                device::convolveL2(
                    dist, idx, prev, input->getGpuMat(stream()), kernel_size, distance_threshold, stream());
                distance_param.updateData(dist, mo::tag::_param = input_param, mo::tag::_context = _ctx.get());
                index_param.updateData(idx, mo::tag::_param = input_param, mo::tag::_context = _ctx.get());
            }
            return true;
        }
    }
}

using namespace aq::pointclouds;
MO_REGISTER_CLASS(ConvolutionL2)
MO_REGISTER_CLASS(ConvolutionL2ForegroundEstimate)
