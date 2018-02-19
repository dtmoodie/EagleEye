#include "IClassifier.hpp"
#include <fstream>

namespace aq
{
    namespace nodes
    {
        void IClassifier::on_label_file_modified(mo::IParam*,
                                                 mo::Context*,
                                                 mo::OptionalTime_t,
                                                 size_t,
                                                 const std::shared_ptr<mo::ICoordinateSystem>&,
                                                 mo::UpdateFlags)
        {
            mo::Mutex_t::scoped_lock lock(getMutex());
            labels = CategorySet(label_file.string());
            BOOST_LOG_TRIVIAL(info) << "Loaded " << labels.size() << " classes";
            labels_param.emitUpdate();
            label_file_param.modified(false);
        }
    }
}
