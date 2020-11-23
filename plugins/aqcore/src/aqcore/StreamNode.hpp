#ifndef AQCORE_STREAM_NODE_HPP
#define AQCORE_STREAM_NODE_HPP
#include <Aquila/nodes/Node.hpp>

namespace aqcore
{
    class AQUILA_EXPORTS StreamNode : virtual public aq::nodes::Node
    {
      public:
        MO_DERIVE(StreamNode, aq::nodes::Node)

        MO_END;

        bool processImpl() override;

        virtual bool processImpl(mo::IAsyncStream& stream) = 0;
        virtual bool processImpl(mo::IDeviceStream& stream) = 0;
    };
} // namespace aqcore
#endif // AQCORE_STREAM_NODE_HPP