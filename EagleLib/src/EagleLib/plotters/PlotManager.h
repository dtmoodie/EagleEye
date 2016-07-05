#pragma once
#include "../Defs.hpp"
#include "Plotter.h"
#include "EagleLib/rcc/shared_ptr.hpp"

namespace EagleLib
{
    class EAGLE_EXPORTS PlotManager
    {
    public:
        static PlotManager& getInstance();
        rcc::shared_ptr<Plotter> getPlot(const std::string& plotName);
        std::vector<std::string> getAvailablePlots();
        std::vector<std::string> getAcceptablePlotters(Parameters::Parameter* param);
        bool canPlotParameter(Parameters::Parameter* param);
    };
}