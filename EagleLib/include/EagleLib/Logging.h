#pragma once
#include "EagleLib/Detail/Export.hpp"
#include <string>
#include <boost/log/trivial.hpp>
//#define LOG(severity) BOOST_LOG(severity) << "[" << __FUNCTION__ << "] " 
namespace EagleLib
{
    void EAGLE_EXPORTS SetupLogging(const std::string& log_dir = "");
    void EAGLE_EXPORTS ShutdownLogging();
}
