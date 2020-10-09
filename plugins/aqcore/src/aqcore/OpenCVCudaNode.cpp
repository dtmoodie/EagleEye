#include "OpenCVCudaNode.hpp"
#include <Aquila/types/CVStream.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
namespace aqcore
{

    std::unique_ptr<cv::cuda::Stream> getCVStream(const mo::IAsyncStreamPtr_t& stream)
    {
        std::shared_ptr<aq::CVStream> cvstream = std::dynamic_pointer_cast<aq::CVStream>(stream);
        std::unique_ptr<cv::cuda::Stream> output;
        if (cvstream)
        {
            output.reset(new cv::cuda::Stream(cvstream->getCVStream()));
        }
        else
        {
            auto cuda_stream = std::dynamic_pointer_cast<mo::cuda::AsyncStream>(stream);
            MO_ASSERT(cuda_stream != nullptr);
            output.reset(new cv::cuda::Stream(cv::cuda::StreamAccessor::wrapStream(*cuda_stream)));
        }
        return output;
    }

    void OpenCVCudaNode::setStream(const mo::IAsyncStreamPtr_t& stream)
    {
        m_stream = aqcore::getCVStream(stream);
        aq::nodes::Node::setStream(stream);
    }

    cv::cuda::Stream& OpenCVCudaNode::getCVStream()
    {
        MO_ASSERT_LOGGER(this->getLogger(), m_stream != nullptr);
        return *m_stream;
    }
} // namespace aqcore