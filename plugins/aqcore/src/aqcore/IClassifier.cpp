#include "IClassifier.hpp"
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>
#include <cereal/types/memory.hpp>
#include <ct/reflect/cerealize.hpp>
#include <fstream>

namespace aqcore
{

    void IClassifier::on_label_file_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
    {
        mo::Mutex_t::Lock_t lock(getMutex());
        std::shared_ptr<aq::CategorySet> labels = std::make_shared<aq::CategorySet>(label_file.string());
        m_labels = labels;
        this->getLogger().info("Loaded {} classes", labels->size());
        this->labels.publish(std::move(labels));
        label_file_param.setModified(false);
    }

    std::shared_ptr<aq::CategorySet> IClassifier::getLabels() const { return m_labels; }

    void IClassifier::setLabels(std::shared_ptr<aq::CategorySet> labels) { m_labels = std::move(labels); }

} // namespace aqcore
