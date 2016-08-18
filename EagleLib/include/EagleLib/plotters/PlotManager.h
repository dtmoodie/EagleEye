#pragma once
#include "EagleLib/Detail/Export.hpp"
#include "Plotter.h"
#include <shared_ptr.hpp>

namespace EagleLib
{
    class EAGLE_EXPORTS PlotManager
    {
    public:
        static PlotManager* Instance();
        rcc::shared_ptr<Plotter> GetPlot(const std::string& plotName);
        std::vector<std::string> GetAvailablePlots();
        std::vector<std::string> GetAcceptablePlotters(mo::IParameter* param);
        bool CanPlotParameter(mo::IParameter* param);
    };
}