#pragma once

#include "IObject.h"
#include "IObjectInfo.h"
#include "ParameteredIObject.h"
#include <shared_ptr.hpp>
#include "EagleLib/Signals.h"
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
#include "SyncedMemory.h"
#include <MetaObject/Signals/detail/SlotMacros.hpp>
#include "MetaObject/Parameters/ParameterMacros.hpp"
#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>

#include <atomic>
#include <string>

RUNTIME_MODIFIABLE_INCLUDE;
RUNTIME_COMPILER_SOURCEDEPENDENCY;
namespace EagleLib
{
    class IDataStream;
    class ICoordinateManager;

    class EAGLE_EXPORTS FrameGrabberInfo: public IObjectInfo
    {
    public:
        /*!
         * \brief GetObjectInfoType indicates that this is a FrameGrabberInfo object
         * \return IObjectInfo::ObjectInfoType::frame_grabber
         */
        virtual int GetObjectInfoType();

        /*!
         * \brief GetObjectName return the factory producible name for this object
         */
        virtual std::string GetObjectName() = 0;
        /*!
         * \brief GetObjectTooltip
         */
        virtual std::string GetObjectTooltip();
        /*!
         * \brief GetObjectHelp return detailed help information for this framegrabber type
         */
        virtual std::string GetObjectHelp();
        /*!
         * \brief CanLoadDocument determines if the frame grabber associated with this info object can load an input document
         * \param document is a string descibing a file / path / URI to load
         * \return 0 if the document cannot be loaded, priority of the frame grabber otherwise.  Higher value means higher compatibility with this document
         */
        virtual int CanLoadDocument(const std::string& document) const = 0;
        /*!
         * \brief LoadTimeout returns the ms that should be allowed for the frame grabber's LoadFile function before a timeout condition
         * \return timeout in ms
         */
        virtual int LoadTimeout() const;

        // Function used for listing what documents are available for loading, used in cases of connected devices to list what
        // devices have been enumerated
        virtual std::vector<std::string> ListLoadableDocuments();
    };
    
    // Interface class for the base level of features frame grabber
    class EAGLE_EXPORTS IFrameGrabber: public TInterface<IID_FrameGrabber, ParameteredIObject>
    {
    public:
        IFrameGrabber();
        virtual bool LoadFile(const std::string& file_path) = 0;
        virtual int GetFrameNumber() = 0;
        virtual int GetNumFrames() = 0;
        virtual std::string GetSourceFilename();
        
        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream) = 0;
        // Get a frame relative to the current frame.  Index can be positive and negative
        virtual TS<SyncedMemory> GetFrameRelative(int index, cv::cuda::Stream& stream) = 0;

        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager() = 0;
        virtual void InitializeFrameGrabber(IDataStream* stream);
        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);

        MO_BEGIN(IFrameGrabber, ParameteredIObject);
            MO_SIGNAL(void, update);
        MO_END;
        
    protected:
        std::string loaded_document;
        IDataStream* parent_stream;
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
        
        virtual int GetFrameNumber();
        
        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrameRelative(int index, cv::cuda::Stream& stream);
        

        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);

        MO_BEGIN(FrameGrabberBuffered, IFrameGrabber)
            PARAM(int, frame_buffer_size, 10);
            PARAM(boost::circular_buffer<TS<SyncedMemory>>, frame_buffer, boost::circular_buffer<TS<SyncedMemory>>());
        MO_END;


    protected:
        virtual void PushFrame(TS<SyncedMemory> frame, bool blocking = true);

        //boost::circular_buffer<TS<SyncedMemory>> frame_buffer;
        
        boost::mutex                             buffer_mtx;
        boost::mutex                             grabber_mtx;
        //std::vector<std::shared_ptr<Signals::connection>> connections;
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
    };
    class EAGLE_EXPORTS FrameGrabberThreaded: public FrameGrabberBuffered
    {
    private:
        void                                     Buffer();
        boost::thread                            buffer_thread;
        
    protected:
        bool _pause = false;
        // Should only ever be called from the buffer thread
        virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream) = 0;
    public:
        virtual ~FrameGrabberThreaded();
        virtual void Init(bool firstInit);
        
        MO_BEGIN(FrameGrabberThreaded, FrameGrabberBuffered);
            MO_SLOT(void, StartThreads);
            MO_SLOT(void, StopThreads);
            MO_SLOT(void, PauseThreads);
            MO_SLOT(void, ResumeThreads);
        MO_END
    };

}
