#pragma once

#include <Aquila/types/ObjectDetection.hpp>

#include "Aquila/nodes/Node.hpp"

#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/types/file_types.hpp"

namespace aqcore
{
    class AQUILA_EXPORTS IClassifier : virtual public aq::nodes::Node
    {
      public:
        MO_DERIVE(IClassifier, aq::nodes::Node)
            PARAM(mo::ReadFile, label_file, {})
            PARAM_UPDATE_SLOT(label_file)

            OUTPUT_WITH_FLAG(std::shared_ptr<aq::CategorySet>, mo::ParamFlags::kUNSTAMPED, labels)
        MO_END;
        std::shared_ptr<aq::CategorySet> getLabels() const;
        void setLabels(std::shared_ptr<aq::CategorySet>);

      private:
        std::shared_ptr<aq::CategorySet> m_labels;
    };

} // namespace aqcore
