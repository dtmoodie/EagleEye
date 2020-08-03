#ifndef AQCORE_OPENCV_CUDA_NODE_HPP
#define AQCORE_OPENCV_CUDA_NODE_HPP
#include <Aquila/nodes/Node.hpp>
#include <Aquila/types/CVStream.hpp>
namespace aqcore
{
    struct OpenCVCudaNode : aq::nodes::Node
    {
        MO_DERIVE(OpenCVCudaNode, aq::nodes::Node)
        MO_END;

        void setStream(const mo::IAsyncStreamPtr_t& stream) override;

        cv::cuda::Stream& getCVStream();
        bool isCVStream() const;

      private:
        std::shared_ptr<aq::CVStream> m_stream;
    };
} // namespace aqcore

#endif // AQCORE_OPENCV_CUDA_NODE_HPP