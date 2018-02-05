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
            labels.clear();
            std::ifstream ifs(label_file.string().c_str());
            if (!ifs)
            {
                MO_LOG_EVERY_N(warning, 100) << "Unable to load label file";
            }

            std::string line;
            while (std::getline(ifs, line, '\n'))
            {
                labels.push_back(line);
            }
            BOOST_LOG_TRIVIAL(info) << "Loaded " << labels.size() << " classes";
            labels_param.emitUpdate();
            label_file_param.modified(false);
        }
    }
}
