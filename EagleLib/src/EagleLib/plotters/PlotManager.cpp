#include "PlotManager.h"
#include "EagleLib/rcc/ObjectManager.h"

using namespace EagleLib;


PlotManager& PlotManager::getInstance()
{
	static PlotManager instance;
	return instance;
}

rcc::shared_ptr<Plotter> PlotManager::getPlot(const std::string& plotName)
{
	
	IObjectConstructor* pConstructor = ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(plotName.c_str());
	if (pConstructor && pConstructor->GetInterfaceId() == IID_Plotter)
	{
		IObject* obj = pConstructor->Construct();
		if (obj)
		{
			obj = obj->GetInterface(IID_Plotter);
			if (obj)
			{
				Plotter* plotter = static_cast<Plotter*>(obj);
				if (plotter)
				{
					plotter->Init(true);
					LOG_TRIVIAL(info) << "[ PlotManager ] successfully generating plot " << plotName;
					return rcc::shared_ptr<Plotter>(plotter);
				}
				else
				{
					LOG_TRIVIAL(warning) << "[ PlotManager ] failed to cast to plotter object " << plotName;
				}
			}
			else
			{
				LOG_TRIVIAL(warning) << "[ PlotManager ] incorrect interface " << plotName;
			}
		}
		else
		{
			LOG_TRIVIAL(warning) << "[ PlotManager ] failed to construct plot " << plotName;
		}
	}
	else
	{
		LOG_TRIVIAL(warning) << "[ PlotManager ] failed to get constructor " << plotName;
	}
	return rcc::shared_ptr<Plotter>();
}

std::vector<std::string> PlotManager::getAvailablePlots()
{
	
	AUDynArray<IObjectConstructor*> constructors;
	ObjectManager::Instance().m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
	std::vector<std::string> output;
	for (size_t i = 0; i < constructors.Size(); ++i)
	{
		if (constructors[i]->GetInterfaceId() == IID_Plotter)
			output.push_back(constructors[i]->GetName());
	}
	return output;
}
