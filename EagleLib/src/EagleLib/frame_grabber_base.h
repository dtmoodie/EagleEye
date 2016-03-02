#pragma once

#include <string>
#include "SyncedMemory.h"
#include "IObjectInfo.h"
#include "IObject.h"
#include "EagleLib/rcc/shared_ptr.hpp"
#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>
#include "ParameteredObject.h"
#include "EagleLib/Signals.h"
#include <atomic>
namespace EagleLib
{
    class DataStream;
    class ICoordinateManager;

    class EAGLE_EXPORTS FrameGrabberInfo: public IObjectInfo
    {
    public:
        virtual int GetObjectInfoType();
        virtual std::string GetObjectName() = 0;
        virtual std::string GetObjectTooltip() = 0;
        virtual std::string GetObjectHelp() = 0;
        virtual bool CanLoadDocument(const std::string& document) const = 0;
        virtual int Priority() const = 0;
        virtual int LoadTimeout() const;
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

        virtual shared_ptr<ICoordinateManager> GetCoordinateManager() = 0;
        virtual void InitializeFrameGrabber(DataStream* stream);

        virtual void Serialize(ISimpleSerializer* pSerializer);
        virtual void Init(bool firstInit);
    protected:
		Signals::typed_signal_base<void()>* update_signal;
        std::string loaded_document;
        DataStream* parent_stream;
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
        ~FrameGrabberBuffered();
        virtual int GetFrameNumber();
        virtual int GetNumFrames() = 0;
        

        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);
        
        virtual void InitializeFrameGrabber(DataStream* stream);
        

        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);
    protected:
        // Should only ever be called from the buffer thread
        virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream) = 0;

        
        void                                     LaunchBufferThread();
        void                                     StopBufferThread();

        boost::circular_buffer<TS<SyncedMemory>> frame_buffer;
        
        boost::mutex                             buffer_mtx;
        boost::mutex                             grabber_mtx;
        //std::vector<std::shared_ptr<Signals::connection>> connections;
        std::atomic_llong                        buffer_begin_frame_number;
        std::atomic_llong                        buffer_end_frame_number;
        std::atomic_llong                        playback_frame_number;
        // If the buffering thread is too far ahead, it will wait on this condition variable
        // until the read thread reads an image from the frame buffer
        boost::condition_variable                  frame_read_cv;
        // If the read thread is too far ahead of the buffer thread, then it will wait on this
        // condition variable for a notification of grabbing of a new image
        boost::condition_variable                  frame_grabbed_cv;
    private:
        void                                     Buffer();
        boost::thread                            buffer_thread;
    };

}