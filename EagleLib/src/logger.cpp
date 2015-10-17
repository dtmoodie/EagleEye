#include "logger.hpp"
#include "nodes/Node.h"

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


using namespace EagleLib;

std::map < const EagleLib::Node*, std::vector < boost::function<void(boost::log::trivial::severity_level, const std::string&)>>> nodeHandlers;
std::vector<boost::function<void(boost::log::trivial::severity_level, const std::string&)>> genericHandlers;


ui_collector::ui_collector(boost::function<void(Node*, const std::string&)> nc, boost::function<void(const std::string&)> gc)
{
    node_callback = nc;
    generic_callback = gc;
}
void ui_collector::consume(boost::log::record_view const& rec, string_type const& message)
{
    auto severity = rec.attribute_values()[boost::log::aux::default_attribute_names::severity()].extract<boost::log::trivial::severity_level>();
    
	if (rec.attribute_values().count("Node"))
	{
        const EagleLib::Node* node = rec.attribute_values()[NodePtr].get();
        boost::log::visit(NodePtr, rec, boost::phoenix::ref(node) == boost::phoenix::placeholders::_1);
        auto& handlers = nodeHandlers[node];
        for (auto handler : handlers)
        {
            handler(severity.get(), message);
        }
	}
	else
	{
        for (auto handler : genericHandlers)
        {
            handler(severity.get(), message);
        }
	}
}
void ui_collector::addNodeCallbackHandler(Node* node, const boost::function<void(boost::log::trivial::severity_level, const std::string&)>& handler)
{
    nodeHandlers[node].push_back(handler);
}
void ui_collector::addGenericCallbackHandler(const boost::function<void(boost::log::trivial::severity_level, const std::string&)>& handler)
{
    genericHandlers.push_back(handler);
}