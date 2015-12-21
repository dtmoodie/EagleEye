#pragma once
#include <vector>
#include <memory>
#include "EagleLib/Defs.hpp"

namespace EagleLib
{
	class IViewManager;
	class ICoordinateManager;
	class IRenderEngine;
    class ITrackManager;
    class DataStreamManager;
	class EAGLE_EXPORTS DataStream
	{
	public:
		// Handles user interactions such as moving the viewport, user interface callbacks, etc
		std::shared_ptr<IViewManager>       view_manager;
		// Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
		std::shared_ptr<ICoordinateManager> coordinate_manager;
		// Handles actual rendering of data.  Use for adding extra objects to the scene
		std::shared_ptr<IRenderEngine>      rendering_engine;
        // Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
        std::shared_ptr<ITrackManager>      track_manager;

        size_t get_stream_id();
        ~DataStream();
    private:
        friend class DataStreamManager;
        DataStream();
        
        size_t stream_id;
	};

    class EAGLE_EXPORTS DataStreamManager
    {
    public:
        static DataStreamManager* instance();
        std::shared_ptr<DataStream> create_stream();
        std::shared_ptr<DataStream> get_stream(size_t id = 0);

    private:
        DataStreamManager();
        ~DataStreamManager();
        std::vector<std::shared_ptr<DataStream>> streams;
    };
}