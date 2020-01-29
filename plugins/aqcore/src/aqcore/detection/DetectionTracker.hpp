#pragma once
#include <Aquila/nodes/Node.hpp>

namespace aq
{
class DetectionTracker : public TInterface<DetectionTracker, nodes::Node>
{
  public:
    template <class T>
    using InterfaceHelper = nodes::Node::InterfaceHelper<T>;

    virtual ~DetectionTracker();

    MO_DERIVE(DetectionTracker, nodes::Node)
        MO_STATIC_SLOT(nodes::Node::Ptr, create, std::string)
        MO_STATIC_SLOT(std::vector<std::string>, list)
    MO_END

  protected:
    bool processImpl() override;
};
}
