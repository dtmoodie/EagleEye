#include "StreamNode.hpp"

namespace aqcore
{
    bool StreamNode::processImpl()
    {
        mo::IAsyncStream::Ptr_t stream = this->getStream();
        if (!stream)
        {
            return false;
        }
        mo::IDeviceStream* dev_stream = stream->getDeviceStream();
        if (dev_stream)
        {
            if (this->processImpl(*dev_stream))
            {
                return true;
            }
        }
        return this->processImpl(*stream);
    }

} // namespace aqcore
