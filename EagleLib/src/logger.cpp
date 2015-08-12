#include "logger.hpp"
#include "nodes/Node.h"

#include <boost/log/utility/value_ref.hpp>
#include <boost/log/utility/formatting_ostream.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/phoenix.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/attributes/current_process_name.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/log/sources/record_ostream.hpp>

using namespace EagleLib;

ui_collector::ui_collector(boost::function<void(Node*, const std::string&)> nc, boost::function<void(const std::string&)> gc):
	node_callback(nc), generic_callback(gc)
{

}
void ui_collector::consume(boost::log::record_view const& rec, string_type const& command_line)
{
	if (rec.attribute_values().count("Node"))
	{
		if (node_callback)
		{
			
			Node* node;
			boost::log::visit(NodePtr, rec, boost::phoenix::ref(node) == boost::phoenix::placeholders::_1);
			//node_callback(node, command_line);
		}
	}
	else
	{
		if (generic_callback)
		{
			//generic_callback(command_line);
		}
	}
}