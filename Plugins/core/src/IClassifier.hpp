#pragma once

#include "Aquila/nodes/Node.hpp"
#include "MetaObject/types/file_types.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include <Aquila/types/ObjectDetection.hpp>
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
                OUTPUT(CategorySet, labels, {})
                APPEND_FLAGS(labels, mo::ParamFlags::Unstamped_e)
            MO_END;
        };
    }
}
