#include "Parameters.h"
#include "Manager.h"
using namespace EagleLib;

Parameter::Ptr Parameter::getParameter(const std::string& fulltreeName)
{
	return EagleLib::NodeManager::getInstance().getParameter(fulltreeName);
}
