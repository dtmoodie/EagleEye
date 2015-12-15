#pragma once
#include <memory>

namespace EagleLib
{
	class IViewManager;
	class ICoordinateManager;
	class IRenderEngine;
    class ITrackManager;
    
	class DataStreamManager
	{
	public:
		DataStreamManager();
		~DataStreamManager();
		// Handles user interactions such as moving the viewport, user interface callbacks, etc
		std::shared_ptr<IViewManager>       view_manager;
		// Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
		std::shared_ptr<ICoordinateManager> coordinate_manager;
		// Handles actual rendering of data.  Use for adding extra objects to the scene
		std::shared_ptr<IRenderEngine>      rendering_engine;
        // Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
        std::shared_ptr<ITrackManager>      track_manager;
	};
}