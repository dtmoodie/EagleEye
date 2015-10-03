#pragma once


#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/function.hpp>

namespace EagleLib
{
	class Node;

	
	class ui_collector : public boost::log::sinks::basic_formatted_sink_backend<char, boost::log::sinks::concurrent_feeding>
	{
		boost::function<void(Node*, const std::string&)> node_callback;
		boost::function<void(const std::string&)> generic_callback;

	public:
		ui_collector(boost::function<void(Node*, const std::string&)> nc = boost::function<void(Node*, const std::string&)>(), boost::function<void(const std::string&)> gc = boost::function<void(const std::string&)>());
		
		void consume(boost::log::record_view const& rec, string_type const& command_line);

	};
}