#include <ct/types/opencv.hpp>

#include "FaceDetector.hpp"
#include <Aquila/nodes/NodeInfo.hpp>

namespace aqcore
{
    aq::nodes::Node::Ptr FaceDetector::create(std::string name)
    {
        // TODO create a child based on name
        auto ctr = mo::MetaObjectFactory::instance()->getConstructor(name.c_str());
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
        auto ctrs = mo::MetaObjectFactory::instance()->getConstructors(FaceDetector::getHash());
        for (auto ctr : ctrs)
        {
            output.push_back(ctr->GetName());
        }
        return output;
    }

    void FaceDetector::createLabels()
    {
        auto labels = std::make_shared<aq::CategorySet>(std::vector<std::string>({"face"}));
        this->labels.publish(labels);
    }

    bool HaarFaceDetector::processImpl()
    {
        auto labels = std::make_shared<aq::CategorySet>(std::vector<std::string>({"face"}));
        this->labels.publish(labels);
        return IImageDetector::processImpl();
    }

} // namespace aqcore
using namespace aqcore;

MO_REGISTER_CLASS(HaarFaceDetector)
