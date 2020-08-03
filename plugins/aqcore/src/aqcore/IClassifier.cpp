#include "IClassifier.hpp"
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>
#include <cereal/types/memory.hpp>
#include <ct/reflect/cerealize.hpp>
#include <fstream>

namespace aq
{
    namespace nodes
    {
        void IClassifier::on_label_file_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
        {
            mo::Mutex_t::Lock_t lock(getMutex());
            std::shared_ptr<CategorySet> labels = std::make_shared<CategorySet>(label_file.string());
            // this->label_file_param.setValue(std::move(labels));
            this->labels.publish(std::move(labels));
            this->getLogger().info("Loaded {} classes", labels->size());
            label_file_param.setModified(false);
        }
    } // namespace nodes
} // namespace aq
