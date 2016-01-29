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
#include <boost/log/attributes/mutable_constant.hpp>


using namespace EagleLib;

std::map < ObjectId, std::vector < boost::function<void(boost::log::trivial::severity_level, const std::string&)>>> nodeHandlers;
std::vector<boost::function<void(boost::log::trivial::severity_level, const std::string&)>> genericHandlers;

boost::log::attributes::mutable_constant<EagleLib::Nodes::Node*> attr(nullptr);
BOOST_LOG_ATTRIBUTE_KEYWORD(NodePtr, "NodePtr", EagleLib::Nodes::Node*);

ui_collector::ui_collector(boost::function<void(Nodes::Node*, const std::string&)> nc, boost::function<void(const std::string&)> gc)
{
    node_callback = nc;
    generic_callback = gc;
    boost::log::core::get()->add_thread_attribute("NodePtr", attr);
}
void ui_collector::consume(boost::log::record_view const& rec, string_type const& message)
{
    auto severity = rec.attribute_values()[boost::log::aux::default_attribute_names::severity()].extract<boost::log::trivial::severity_level>();
    int count = rec.attribute_values().count("NodePtr");
	if (count)
	{
        //auto node = rec.attribute_values()[NodePtr].get();
        auto node = rec.attribute_values()[NodePtr].get();
        if (node != nullptr)
        {
            auto& handlers = nodeHandlers[node->GetObjectId()];
            for (auto handler : handlers)
            {
                handler(severity.get(), message);
            }
			return;
		}
	}
	for (auto handler : genericHandlers)
	{
		handler(severity.get(), message);
	}
}
size_t ui_collector::addNodeCallbackHandler(Nodes::Node* node, const boost::function<void(boost::log::trivial::severity_level, const std::string&)>& handler)
{
    nodeHandlers[node->GetObjectId()].push_back(handler);
	return nodeHandlers[node->GetObjectId()].size() - 1;
}
void ui_collector::removeNodeCallbackHandler(Nodes::Node* node, size_t id)
{
	auto& handlers = nodeHandlers[node->GetObjectId()];
	handlers.erase(handlers.begin() + id);
	if (handlers.empty())
	{
		auto itr = nodeHandlers.find(node->GetObjectId());
		nodeHandlers.erase(itr);
	}
}
size_t ui_collector::addGenericCallbackHandler(const boost::function<void(boost::log::trivial::severity_level, const std::string&)>& handler)
{
    genericHandlers.push_back(handler);
	return genericHandlers.size() - 1;
}
void ui_collector::clearGenericCallbackHandlers()
{
	genericHandlers.clear();
}
void ui_collector::setNode(EagleLib::Nodes::Node* node)
{
    attr.set(node);
}
EagleLib::Nodes::Node* ui_collector::getNode()
{
	return attr.get();
}