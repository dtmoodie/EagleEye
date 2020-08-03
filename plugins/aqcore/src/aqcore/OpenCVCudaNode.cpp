#include "OpenCVCudaNode.hpp"

namespace aqcore
{
    void OpenCVCudaNode::setStream(const mo::IAsyncStreamPtr_t& stream)
    {
        m_stream = std::dynamic_pointer_cast<aq::CVStream>(stream);
        if (!m_stream)
        {
            this->getLogger().info("Setting stream to a non CVStream");
        }
        aq::nodes::Node::setStream(stream);
    }

    cv::cuda::Stream& OpenCVCudaNode::getCVStream()
    {
        MO_ASSERT_LOGGER(this->getLogger(), m_stream != nullptr);
        return m_stream->getCVStream();
    }

    bool OpenCVCudaNode::isCVStream() const { return m_stream != nullptr; }
} // namespace aqcore