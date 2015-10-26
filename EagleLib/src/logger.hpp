#pragma once


#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/function.hpp>
#include <EagleLib/Defs.hpp>
#include <boost/log/trivial.hpp>
namespace EagleLib
{
	class Node;

	
    class EAGLE_EXPORTS ui_collector : public boost::log::sinks::basic_formatted_sink_backend<char, boost::log::sinks::concurrent_feeding>
	{
		boost::function<void(Node*, const std::string&)> node_callback;
		boost::function<void(const std::string&)> generic_callback;

	public:
		ui_collector(boost::function<void(Node*, const std::string&)> nc = boost::function<void(Node*, const std::string&)>(), boost::function<void(const std::string&)> gc = boost::function<void(const std::string&)>());
		
		void consume(boost::log::record_view const& rec, string_type const& command_line);
        void setNodeCallback(boost::function<void(Node*, const std::string&)> nc);
        void setGenericCallback(boost::function<void(const std::string&)> gc);
        static void addNodeCallbackHandler(Node* node, const boost::function<void(boost::log::trivial::severity_level, const std::string&)>& handler);
        static size_t addGenericCallbackHandler(const boost::function<void(boost::log::trivial::severity_level, const std::string&)>& handler);
		static void clearGenericCallbackHandlers();
        static void setNode(EagleLib::Node* node);
		static EagleLib::Node* getNode();
	};
}