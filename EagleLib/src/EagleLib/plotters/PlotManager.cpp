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
                    LOG(info) << "[ PlotManager ] successfully generating plot " << plotName;
                    return rcc::shared_ptr<Plotter>(plotter);
                }
                else
                {
                    LOG(warning) << "[ PlotManager ] failed to cast to plotter object " << plotName;
                }
            }
            else
            {
                LOG(warning) << "[ PlotManager ] incorrect interface " << plotName;
            }
        }
        else
        {
            LOG(warning) << "[ PlotManager ] failed to construct plot " << plotName;
        }
    }
    else
    {
        LOG(warning) << "[ PlotManager ] failed to get constructor " << plotName;
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
std::vector<std::string> PlotManager::getAcceptablePlotters(Parameters::Parameter* param)
{
    auto constructors = ObjectManager::Instance().GetConstructorsForInterface(IID_Plotter);
    std::vector<std::string> output;
    for(auto& constructor : constructors)
    {
        auto object_info = constructor->GetObjectInfo();
        if(object_info)
        {
            auto plot_info = dynamic_cast<PlotterInfo*>(object_info);
            if(plot_info)
            {
                if(plot_info->AcceptsParameter(param))
                {
                    output.push_back(constructor->GetName());
                }
            }
        }
    }
    return output;
}