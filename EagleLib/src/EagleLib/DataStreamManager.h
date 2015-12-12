#pragma once
#include <memory>


namespace EagleLib
{
	class IViewManager;
	class ICoordinateManager;
	class IRenderEngine;
    
	class DataStreamManager
	{
	public:
		std::shared_ptr<IViewManager>       view_manager;
		std::shared_ptr<ICoordinateManager> coordinate_manager;
		std::shared_ptr<IRenderEngine>      rendering_engine;
	};
}