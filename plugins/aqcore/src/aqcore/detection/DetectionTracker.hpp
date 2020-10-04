#pragma once
#include <Aquila/nodes/Node.hpp>

namespace aqcore
{
    class DetectionTracker : public TInterface<DetectionTracker, aq::nodes::Node>
    {
      public:
        template <class T>
        using InterfaceHelper = aq::nodes::Node::InterfaceHelper<T>;

        virtual ~DetectionTracker();

        MO_DERIVE(DetectionTracker, aq::nodes::Node)
            MO_STATIC_SLOT(aq::nodes::Node::Ptr, create, std::string)
            MO_STATIC_SLOT(std::vector<std::string>, list)
        MO_END;

      protected:
        bool processImpl() override;
    };
} // namespace aqcore
