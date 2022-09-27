#ifndef AQCORE_STREAM_NODE_HPP
#define AQCORE_STREAM_NODE_HPP
#include <Aquila/nodes/Node.hpp>

namespace aqcore
{
    class AQUILA_EXPORTS StreamNode : virtual public aq::nodes::GPUNode
    {
      public:
        MO_DERIVE(StreamNode, aq::nodes::GPUNode)

        MO_END;

        bool processImpl() override;

        bool processImpl(mo::IAsyncStream& stream) override = 0;
        bool processImpl(mo::IDeviceStream& stream) override = 0;
    };
} // namespace aqcore
#endif // AQCORE_STREAM_NODE_HPP
