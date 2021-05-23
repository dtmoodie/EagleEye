#include <Aquila/types/SyncedImage.hpp>

#include "l2_convolution.hpp"

#include <Aquila/nodes/NodeInfo.hpp>

#include <RuntimeObjectSystem/RuntimeSourceDependency.h>

RUNTIME_COMPILER_SOURCEDEPENDENCY_EXT(".cu")

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
        } // namespace device

        bool ConvolutionL2::processImpl()
        {
            cv::cuda::GpuMat dist, idx;
            auto stream = this->getStream();
            mo::IDeviceStream* dev_stream = stream->getDeviceStream();
            cv::cuda::GpuMat input_mat = input->getGpuMat(dev_stream);
            cv::cuda::Stream& cvstream = this->getCVStream();
            device::convolveL2(dist, idx, input_mat, kernel_size, distance_threshold, cvstream);
            distance.publish(dist, mo::tags::param = &input_param, mo::tags::stream = stream.get());
            index.publish(idx, mo::tags::param = &input_param, mo::tags::stream = stream.get());
            return true;
        }

        void ConvolutionL2ForegroundEstimate::buildModel() { this->build_model = true; }

        bool ConvolutionL2ForegroundEstimate::processImpl()
        {
            auto stream = this->getStream();
            if (build_model)
            {
                prev = input->getGpuMat();
                build_model = false;
            }
            if (!prev.empty())
            {
                cv::cuda::GpuMat dist, idx;
                mo::IDeviceStream* dev_stream = stream->getDeviceStream();
                cv::cuda::GpuMat in_mat = input->getGpuMat(dev_stream);
                cv::cuda::Stream& cvstream = this->getCVStream();
                device::convolveL2(dist, idx, prev, in_mat, kernel_size, distance_threshold, cvstream);

                distance.publish(dist, mo::tags::param = &input_param, mo::tags::stream = stream.get());
                index.publish(idx, mo::tags::param = &input_param, mo::tags::stream = stream.get());
            }
            return true;
        }
    } // namespace pointclouds
} // namespace aq

using namespace aq::pointclouds;
MO_REGISTER_CLASS(ConvolutionL2)
MO_REGISTER_CLASS(ConvolutionL2ForegroundEstimate)
