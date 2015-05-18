#include "Parameters.h"
#include "Manager.h"
using namespace EagleLib;

Parameter::Ptr Parameter::getParameter(const std::string& fullTreeName)
{
	return EagleLib::NodeManager::getInstance().getParameter(fullTreeName);
}
