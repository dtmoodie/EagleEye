#include "EagleLib/logger.hpp"
#include "EagleLib/Nodes/Node.h"

#include <boost/log/utility/value_ref.hpp>
#include <boost/log/utility/formatting_ostream.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/phoenix.hpp>

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/attributes/current_process_name.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/attributes/value_extraction.hpp>
#include <boost/log/attributes/mutable_constant.hpp>


using namespace EagleLib;

boost::log::attributes::mutable_constant<std::string>    attr(std::string(""));

BOOST_LOG_ATTRIBUTE_KEYWORD(NodeName,"NodeName",std::string);

std::map<std::string, std::shared_ptr<mo::TypedSignal<void(boost::log::trivial::severity_level, std::string)>>> object_specific_signals;
mo::TypedSignal<void(boost::log::trivial::severity_level, std::string)>                                         generic_signal;
mo::TypedSignal<void(boost::log::trivial::severity_level, std::string, std::string)>                            object_signal;

ui_collector::ui_collector(object_log_handler_t olh, log_handler_t lh)
{
    object_log_handler = olh;
    log_handler = lh;
    boost::log::core::get()->add_thread_attribute("NodeName", attr);
}
void ui_collector::consume(boost::log::record_view const& rec, string_type const& message)
{
    auto severity = rec.attribute_values()[boost::log::aux::default_attribute_names::severity()].extract<boost::log::trivial::severity_level>();
    
    if (rec.attribute_values().count("NodeName"))
    {
        auto name = rec.attribute_values()[NodeName].get();
        if(name.size())
        {
            auto itr = object_specific_signals.find(name);
            if(itr != object_specific_signals.end())
            {
                if (itr->second)
                {
                    (*itr->second)(severity.get(), message);
                }
            }
            object_signal(severity.get(), name, message);
            return;
        }
    }
    generic_signal(severity.get(), message);
}


mo::TypedSignal<void(boost::log::trivial::severity_level, std::string)>& ui_collector::get_object_log_handler(std::string name)
{
    auto& sig = object_specific_signals[name];
    if(!sig)
        sig.reset(new mo::TypedSignal<void(boost::log::trivial::severity_level, std::string)>());
    return (*sig);
}
mo::TypedSignal<void(boost::log::trivial::severity_level, std::string, std::string)>& ui_collector::get_object_log_handler()
{
    return object_signal;
}
mo::TypedSignal<void(boost::log::trivial::severity_level, std::string)>& ui_collector::get_log_handler()
{
    return generic_signal;
}

void ui_collector::set_node_name(std::string name)
{
    attr.set(name);
}

