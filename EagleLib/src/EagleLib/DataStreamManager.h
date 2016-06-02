#pragma once
#include <vector>
#include <memory>

#include "EagleLib/Defs.hpp"
#include "EagleLib/rcc/shared_ptr.hpp"
#include "EagleLib/ParameteredIObject.h"
#include "EagleLib/frame_grabber_base.h"
#include "EagleLib/nodes/Node.h"
#include "IViewManager.h"
#include "ICoordinateManager.h"
#include "rendering/RenderingEngine.h"
#include "tracking/ITrackManager.h"



#include <opencv2/core/cuda.hpp>
#include <boost/thread.hpp>

namespace Parameters
{
	class IVariableManager;
}

namespace EagleLib
{
	namespace Nodes
	{
		class Node;
	}
	class IViewManager;
	class ICoordinateManager;
	class IRenderEngine;
    class ITrackManager;
    class DataStreamManager;
    class IFrameGrabber;
    class SignalManager;
	class IParameterBuffer;
    class IVariableSink;

	class EAGLE_EXPORTS DataStream: public TInterface<IID_DataStream, ParameteredIObject>
	{
	public:
        typedef rcc::shared_ptr<DataStream> Ptr;
        static bool CanLoadDocument(const std::string& document);
		DataStream();
        ~DataStream();

        // Handles user interactions such as moving the viewport, user interface callbacks, etc
        rcc::shared_ptr<IViewManager>            GetViewManager();

        // Handles conversion of coordinate systems, such as to and from image coordinates, world coordinates, render scene coordinates, etc.
        rcc::shared_ptr<ICoordinateManager>      GetCoordinateManager();
        
        // Handles actual rendering of data.  Use for adding extra objects to the scene
        rcc::shared_ptr<IRenderEngine>           GetRenderingEngine();
        
        // Handles tracking objects within a stream and communicating with the global track manager to track across multiple data streams
        rcc::shared_ptr<ITrackManager>            GetTrackManager();

        // Handles actual loading of the image, etc
        rcc::shared_ptr<IFrameGrabber>           GetFrameGrabber();

		Parameters::IVariableManager*		GetVariableManager();

        SignalManager*							GetSignalManager();

		IParameterBuffer*						GetParameterBuffer();

        std::vector<rcc::shared_ptr<Nodes::Node>> GetNodes();

        bool LoadDocument(const std::string& document);
        
        int get_stream_id();

        void AddNode(rcc::shared_ptr<Nodes::Node> node);
        void AddNodes(std::vector<rcc::shared_ptr<Nodes::Node>> node);
        void RemoveNode(rcc::shared_ptr<Nodes::Node> node);
        
        void StartThread();
        void StopThread();
        void PauseThread();
        void ResumeThread();
        void process();
        
        void AddVariableSink(IVariableSink* sink);
        void RemoveVariableSink(IVariableSink* sink);

    protected:
        friend class DataStreamManager;
        // members
        int stream_id;
		size_t _thread_id;
        rcc::shared_ptr<IViewManager>							view_manager;
        rcc::shared_ptr<ICoordinateManager>						coordinate_manager;
        rcc::shared_ptr<IRenderEngine>							rendering_engine;
        rcc::shared_ptr<ITrackManager>							track_manager;
        rcc::shared_ptr<IFrameGrabber>							frame_grabber;
		std::shared_ptr<Parameters::IVariableManager>			variable_manager;
		std::shared_ptr<SignalManager>							signal_manager;
        std::vector<rcc::shared_ptr<Nodes::Node>>				top_level_nodes;
		std::shared_ptr<IParameterBuffer>						_parameter_buffer;
        std::mutex    											nodes_mtx;
        bool													paused;
        cv::cuda::Stream										cuda_stream;
        boost::thread											processing_thread;
        volatile bool											dirty_flag;
        std::vector<std::shared_ptr<Signals::connection>>		connections;
		cv::cuda::Stream										streams[2];
        std::vector<IVariableSink*>                             variable_sinks;
    public:
		SIGNALS_BEGIN(DataStream)
			SIG_SEND(StartThreads);
			SIG_SEND(StopThreads);
		SIGNALS_END
	};

    class EAGLE_EXPORTS DataStreamManager
    {
    public:
        static DataStreamManager* instance();
        rcc::shared_ptr<DataStream> create_stream();
        DataStream* get_stream(size_t id = 0);
        void destroy_stream(DataStream* stream);

    private:
        DataStreamManager();
        ~DataStreamManager();

        std::vector<rcc::shared_ptr<DataStream>>    streams;    
    };
}
