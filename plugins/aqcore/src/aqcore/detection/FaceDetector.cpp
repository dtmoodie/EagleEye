#include "FaceDetector.hpp"
#include <Aquila/nodes/NodeInfo.hpp>
namespace aq
{
namespace nodes
{
Node::Ptr FaceDetector::create(std::string name)
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

std::vector<std::string> FaceDetector::list()
{
    // TODO query all children of this class
    std::vector<std::string> output;
    auto ctrs = mo::MetaObjectFactory::instance().getConstructors(FaceDetector::getHash());
    for (auto ctr : ctrs)
    {
        output.push_back(ctr->GetName());
    }
    return output;
}

void FaceDetector::createLabels()
{
    if (labels->size() == 0)
    {
        labels = std::make_shared<aq::CategorySet>(std::vector<std::string>({"face"}));
    }
}

bool FaceDetector::processImpl()
{
    return false;
}

#if MO_OPENCV_HAVE_CUDA == 1
bool HaarFaceDetector::processImpl()
{
    createLabels();
    return HaarDetector::processImpl();
}
#endif

}
}
using namespace aq::nodes;


MO_REGISTER_CLASS(FaceDetector)
#if MO_OPENCV_HAVE_CUDA == 1
MO_REGISTER_CLASS(HaarFaceDetector)
#endif
