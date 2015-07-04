#include "Parameters.hpp"

using namespace Parameters;
Parameter::Parameter(const std::string& name_, const ParameterTypes& type_, const std::string& tooltip_):
	name(name_), type(type_), tooltip(tooltip_)
{
}
std::string& Parameter::GetName()
{
	return name;
}
void Parameter::SetName(const std::string& name_)
{
	name = name_;
}
const std::string& Parameter::GetTooltip()
{
	return tooltip;
}
void Parameter::SetTooltip(const std::string& tooltip_)
{
	tooltip = tooltip_;
}
const std::string& Parameter::GetTreeName()
{
	return treeName;
}
void Parameter::SetTreeName(const std::string& treeName_)
{
	treeName = treeName_;
}
boost::signals2::connection Parameter::RegisterNotifier(const boost::function<void(void)>& f)
{
	return UpdateSignal.connect(f);
}