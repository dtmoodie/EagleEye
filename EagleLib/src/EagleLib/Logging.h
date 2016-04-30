#pragma once
#include <EagleLib/Defs.hpp>
#include <string>
#include <boost/log/trivial.hpp>
//#define LOG(severity) BOOST_LOG(severity) << "[" << __FUNCTION__ << "] " 
namespace EagleLib
{
	void EAGLE_EXPORTS SetupLogging();
	void EAGLE_EXPORTS ShutdownLogging();
}