#include "DetectionTracker.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

namespace aq
{

nodes::Node::Ptr DetectionTracker::create(std::string name)
{
    // TODO create a child based on name
    auto ctr = mo::MetaObjectFactory::instance().getConstructor(name.c_str());
    if (ctr)
    {
        auto obj = ctr->Construct();
        if (obj)
        {
            obj->Init(true);
            return {obj};
        }
    }
    return {};
}

std::vector<std::string> DetectionTracker::list()
{
    // TODO query all children of this class
    std::vector<std::string> output;
    auto ctrs = mo::MetaObjectFactory::instance().getConstructors(DetectionTracker::getHash());
    for (auto ctr : ctrs)
    {
        output.push_back(ctr->GetName());
    }
    return output;
}

DetectionTracker::~DetectionTracker()
{
}

bool DetectionTracker::processImpl()
{
    return false;
}

}
using namespace aq;
MO_REGISTER_CLASS(DetectionTracker)
