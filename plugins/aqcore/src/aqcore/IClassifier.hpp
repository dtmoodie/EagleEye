#pragma once

#include <Aquila/types/ObjectDetection.hpp>

#include "Aquila/nodes/Node.hpp"

#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/types/file_types.hpp"

namespace aq
{
    namespace nodes
    {
        class AQUILA_EXPORTS IClassifier : public Node
        {
          public:
            MO_DERIVE(IClassifier, Node)
                PARAM(mo::ReadFile, label_file, {})
                PARAM_UPDATE_SLOT(label_file)

                OUTPUT_WITH_FLAG(std::shared_ptr<CategorySet>, mo::ParamFlags::kUNSTAMPED, labels)
            MO_END;
        };
    } // namespace nodes
} // namespace aq
