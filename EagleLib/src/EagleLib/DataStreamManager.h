#pragma once
#include <vector>
#include <memory>

#include "EagleLib/Defs.hpp"
#include "EagleLib/rcc/shared_ptr.hpp"

#include "IViewManager.h"
#include "ICoordinateManager.h"
#include "rendering/RenderingEngine.h"
#include "tracking/ITrackManager.h"
#include "frame_grabber_base.h"
#include "nodes/Node.h"
#include "EagleLib/Signals.h"


namespace EagleLib
{
	class IViewManager;
	class ICoordinateManager;
	class IRenderEngine;
    class ITrackManager;
    class DataStreamManager;
    class IFrameGrabber;
    class SignalManager;
	class IVariableManager;

	class EAGLE_EXPORTS DataStream
	{
	public:
        typedef std::shared_ptr<DataStream> Ptr;
        static bool CanLoadDocument(const std::string& document);

        ~DataStream();

        // Handles user interactions such as moving the viewport, user interface callbacks, etc
        shared_ptr<IViewManager>            GetViewManager();

        // Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
        shared_ptr<ICoordinateManager>      GetCoordinateManager();
        
        // Handles actual rendering of data.  Use for adding extra objects to the scene
        shared_ptr<IRenderEngine>           GetRenderingEngine();
        
        // Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
        shared_ptr<ITrackManager>            GetTrackManager();

        // Handles actual loading of the image, etc
        shared_ptr<IFrameGrabber>           GetFrameGrabber();

		std::shared_ptr<IVariableManager>   GetVariableManager();

        SignalManager*                      GetSignalManager();
        std::vector<shared_ptr<Nodes::Node>> GetNodes();

        bool LoadDocument(const std::string& document);
        

        int get_stream_id();

        void AddNode(shared_ptr<Nodes::Node> node);
        void AddNode(std::vector<shared_ptr<Nodes::Node>> node);
        void RemoveNode(shared_ptr<Nodes::Node> node);
        
        void LaunchProcess();
        void StopProcess();
        void PauseProcess();
        void ResumeProcess();
        void process();
    
        
    protected:
        friend class DataStreamManager;

        DataStream();
        

        // members
        int stream_id;
        shared_ptr<IViewManager>							view_manager;
        shared_ptr<ICoordinateManager>						coordinate_manager;
        shared_ptr<IRenderEngine>							rendering_engine;
        shared_ptr<ITrackManager>							track_manager;
        shared_ptr<IFrameGrabber>							frame_grabber;
		std::shared_ptr<IVariableManager>					variable_manager;
		std::shared_ptr<SignalManager>						signal_manager;
        std::vector<shared_ptr<Nodes::Node>>				top_level_nodes;
        std::mutex											nodes_mtx;
        bool												paused;
        cv::cuda::Stream									cuda_stream;
        boost::thread										processing_thread;
        volatile bool										dirty_flag;
        std::vector<std::shared_ptr<Signals::connection>>	connections;
	};

    class EAGLE_EXPORTS DataStreamManager
    {
    public:
        static DataStreamManager* instance();
        std::shared_ptr<DataStream> create_stream();
        DataStream* get_stream(size_t id = 0);
        void destroy_stream(DataStream* stream);

    private:
        DataStreamManager();
        ~DataStreamManager();

        std::vector<std::shared_ptr<DataStream>>    streams;    
    };
}