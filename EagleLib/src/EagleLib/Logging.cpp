#include "Logging.h"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/exceptions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/expressions/attr.hpp>
#include <boost/log/attributes/time_traits.hpp>
#include <boost/log/expressions/formatters.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/expressions/formatters/named_scope.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/sinks/async_frontend.hpp>
#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/filesystem.hpp>
#include <EagleLib/logger.hpp>
boost::shared_ptr< boost::log::sinks::asynchronous_sink<EagleLib::ui_collector>> log_sink;
void EagleLib::SetupLogging()
{
#ifdef _DEBUG
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
#else
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
#endif
	boost::log::add_common_attributes();
	if (!boost::filesystem::exists("./logs") || !boost::filesystem::is_directory("./logs"))
	{
		boost::filesystem::create_directory("./logs");
	}
	boost::log::core::get()->add_global_attribute("Scope", boost::log::attributes::named_scope());
	// https://gist.github.com/xiongjia/e23b9572d3fc3d677e3d

	auto consoleFmtTimeStamp = boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%M:%S.%f");

	auto fmtThreadId = boost::log::expressions::attr<boost::log::attributes::current_thread_id::value_type>("ThreadID");

	auto fmtSeverity = boost::log::expressions::attr<boost::log::trivial::severity_level>("Severity");

	auto fmtScope = boost::log::expressions::format_named_scope("Scope",
		boost::log::keywords::format = "%n(%f:%l)",
		boost::log::keywords::iteration = boost::log::expressions::reverse,
		boost::log::keywords::depth = 2);

	boost::log::formatter consoleFmt =
		boost::log::expressions::format("%1%<%2%,%3%> %4%")
		% consoleFmtTimeStamp					// 1
		% fmtThreadId							// 2
		% fmtSeverity							// 3
		% boost::log::expressions::smessage;	// 4

	auto consoleSink = boost::log::add_console_log(std::clog);
	consoleSink->set_formatter(consoleFmt);


	auto fmtTimeStamp = boost::log::expressions::
		format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S.%f");

	// File sink 
	boost::log::formatter logFmt =
		boost::log::expressions::format("[%1%] (%2%) [%3%] [%4%] %5%")
		% fmtTimeStamp
		% fmtThreadId
		% fmtSeverity
		% fmtScope
		% boost::log::expressions::smessage;


	auto fsSink = boost::log::add_file_log(
		boost::log::keywords::file_name = "./logs/%Y-%m-%d_%H-%M-%S.%N.log",
		boost::log::keywords::rotation_size = 10 * 1024 * 1024,
		boost::log::keywords::min_free_space = 30 * 1024 * 1024,
		boost::log::keywords::open_mode = std::ios_base::app);
	fsSink->set_formatter(logFmt);
	fsSink->locked_backend()->auto_flush(true);
	
	log_sink.reset(new boost::log::sinks::asynchronous_sink<EagleLib::ui_collector>());

	boost::log::core::get()->add_sink(log_sink);
}
void EagleLib::ShutdownLogging()
{
	log_sink->flush();
	log_sink->stop();
	log_sink.reset();
}