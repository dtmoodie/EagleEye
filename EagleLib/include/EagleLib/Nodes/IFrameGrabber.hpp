#pragma once
#include "EagleLib/SyncedMemory.h"
#include "EagleLib/Nodes/Node.h"
#include "EagleLib/Nodes/NodeInfo.hpp"
#include "IObject.h"
#include "IObjectInfo.h"


#include <MetaObject/Signals/detail/SlotMacros.hpp>
#include <MetaObject/Parameters/ParameterMacros.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Signals/detail/SignalMacros.hpp>
#include <MetaObject/Context.hpp>
#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/Thread/ThreadHandle.hpp>
#include <MetaObject/Thread/ThreadPool.hpp>

#include <RuntimeInclude.h>
#include <RuntimeSourceDependency.h>
#include <shared_ptr.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>


#include <atomic>
#include <string>

RUNTIME_MODIFIABLE_INCLUDE;
RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("../../../src/EagleLib/Nodes/IFrameGrabber", ".cpp");
namespace EagleLib
{
    namespace Nodes
    {
        class IFrameGrabber;
        class FrameGrabberInfo;
    }
}



namespace EagleLib
{
    class IDataStream;
    class ICoordinateManager;
    namespace Nodes
    {
    
    class EAGLE_EXPORTS FrameGrabberInfo: virtual public NodeInfo
    {
    public:
        /*!
         * \brief CanLoadDocument determines if the frame grabber associated with this info object can load an input document
         * \param document is a string descibing a file / path / URI to load
         * \return 0 if the document cannot be loaded, priority of the frame grabber otherwise.  Higher value means higher compatibility with this document
         */
        virtual int CanLoadDocument(const ::std::string& document) const = 0;
        /*!
         * \brief LoadTimeout returns the ms that should be allowed for the frame grabber's LoadFile function before a timeout condition
         * \return timeout in ms
         */
        virtual int LoadTimeout() const;

        // Function used for listing what documents are available for loading, used in cases of connected devices to list what
        // devices have been enumerated
        virtual ::std::vector<::std::string> ListLoadableDocuments() const;

        ::std::string Print() const;
    };


    
    // Interface class for the base level of features frame grabber
    class EAGLE_EXPORTS IFrameGrabber: public TInterface<ctcrc32("EagleLib::Nodes::IFrameGrabber"), Node>
    {
    public:
        typedef FrameGrabberInfo InterfaceInfo;
        typedef IFrameGrabber Interface;

        static rcc::shared_ptr<IFrameGrabber> Create(const ::std::string& uri, const ::std::string& preferred_loader = "");
        static ::std::vector<::std::string> ListAllLoadableDocuments();

        IFrameGrabber();
        virtual bool LoadFile(const ::std::string& file_path) = 0;
        virtual long long GetFrameNumber() = 0;
        virtual long long GetNumFrames() = 0;
        virtual ::std::string GetSourceFilename();
        
        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream) = 0;
        // Get a frame relative to the current frame.  Index can be positive and negative
        virtual TS<SyncedMemory> GetFrameRelative(int index, cv::cuda::Stream& stream) = 0;

        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager() = 0;
        virtual void InitializeFrameGrabber(IDataStream* stream);
        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);

        
        MO_DERIVE(IFrameGrabber, Node)
            MO_SIGNAL(void, update)
            MO_SLOT(void, Restart)
            OUTPUT(SyncedMemory, current_frame, SyncedMemory())
            PARAM(std::string, loaded_document, "")
            MO_SLOT(void, on_loaded_document_modified, mo::Context*, mo::IParameter*)
        MO_END
        
    protected:
        bool ProcessImpl();

        IFrameGrabber(const IFrameGrabber&) = delete;
        IFrameGrabber& operator=(const IFrameGrabber&) = delete;
        IDataStream* parent_stream;
        //cv::cuda::Stream stream;
        //mo::Context ctx;
    };
    //   [ 0 ,1, 2, 3, 4, 5 ....... N-5, N-4, N-3, N-2, N-1, N]
    //    buffer begin                                  buffer end
    //            |      safe playback frames          |
    //        buffer begin + 5 < playback            > buffer end - 5
    //                    
    class EAGLE_EXPORTS FrameGrabberBuffered: public IFrameGrabber
    {
    public:
        FrameGrabberBuffered();
        virtual ~FrameGrabberBuffered();
        
        virtual long long GetFrameNumber();
        
        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);

        SyncedMemory get_frame(int ts, cv::cuda::Stream& stream);

        MO_DERIVE(FrameGrabberBuffered, IFrameGrabber)
            PARAM(int, frame_buffer_size, 10)
            OUTPUT(boost::circular_buffer<TS<SyncedMemory>>, frame_buffer, boost::circular_buffer<TS<SyncedMemory>>())
            MO_SLOT(TS<SyncedMemory>, GetFrame, int, cv::cuda::Stream&)
            MO_SLOT(TS<SyncedMemory>, GetNextFrame, cv::cuda::Stream&)
            MO_SLOT(TS<SyncedMemory>, GetFrameRelative, int, cv::cuda::Stream&)
        MO_END
    protected:
        virtual void PushFrame(TS<SyncedMemory> frame, bool blocking = true);
        boost::mutex                             buffer_mtx;
        boost::mutex                             grabber_mtx;
        std::atomic_llong                        buffer_begin_frame_number;
        std::atomic_llong                        buffer_end_frame_number;
        std::atomic_llong                        playback_frame_number;
        // If the buffering thread is too far ahead, it will wait on this condition variable
        // until the read thread reads an image from the frame buffer
        boost::condition_variable                frame_read_cv;
        // If the read thread is too far ahead of the buffer thread, then it will wait on this
        // condition variable for a notification of grabbing of a new image
        boost::condition_variable                frame_grabbed_cv;
        bool                                     _is_stream;
        std::queue<long long>        _frame_number_playback_queue;
    };
    class EAGLE_EXPORTS FrameGrabberThreaded: public FrameGrabberBuffered
    {
    public:
        FrameGrabberThreaded();
        MO_DERIVE(FrameGrabberThreaded, FrameGrabberBuffered)
            MO_SLOT(void, StartThreads)
            MO_SLOT(void, StopThreads)
            MO_SLOT(void, PauseThreads)
            MO_SLOT(void, ResumeThreads)
            MO_SLOT(int, Buffer)
            PROPERTY(mo::ThreadHandle, _buffer_thread_handle, mo::ThreadPool::Instance()->RequestThread())
            MO_SIGNAL(void, eos)
        MO_END
            void Init(bool firstInit);
    protected:
        // Should only ever be called from the buffer thread
        virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream) = 0;
        long long _empty_frame_count;
    };
    }
}
