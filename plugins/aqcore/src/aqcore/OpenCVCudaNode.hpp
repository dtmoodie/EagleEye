#ifndef AQCORE_OPENCV_CUDA_NODE_HPP
#define AQCORE_OPENCV_CUDA_NODE_HPP
#include <Aquila/nodes/Node.hpp>

namespace cv
{
    namespace cuda
    {
        class Stream;
    }
} // namespace cv

namespace aqcore
{

    AQUILA_EXPORTS std::unique_ptr<cv::cuda::Stream> getCVStream(const mo::IAsyncStreamPtr_t& stream);
    struct OpenCVCudaNode : virtual aq::nodes::Node
    {
        MO_DERIVE(OpenCVCudaNode, aq::nodes::Node)
        MO_END;

        void setStream(const mo::IAsyncStreamPtr_t& stream) override;

        cv::cuda::Stream& getCVStream();

      private:
        std::unique_ptr<cv::cuda::Stream> m_stream;
    };
} // namespace aqcore

#endif // AQCORE_OPENCV_CUDA_NODE_HPP