#pragma once
#include "Defs.hpp"
#include "plotters/Plotter.h"
#include "shared_ptr.hpp"

namespace EagleLib
{
	class EAGLE_EXPORTS PlotManager
	{
	public:
		static PlotManager& getInstance();
		shared_ptr<Plotter> getPlot(const std::string& plotName);
		std::vector<std::string> getAvailablePlots();
	};
}