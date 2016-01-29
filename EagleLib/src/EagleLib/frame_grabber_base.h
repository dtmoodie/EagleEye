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
        int LoadTimeout() const;
        
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
    protected:
        Signals::signal<void()>* update_signal;
        std::string loaded_document;
    };


    class EAGLE_EXPORTS FrameGrabberBuffered: public IFrameGrabber
    {
    public:
        FrameGrabberBuffered();
        virtual int GetFrameNumber();
        virtual int GetNumFrames() = 0;
        

        virtual TS<SyncedMemory> GetCurrentFrame(cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetFrame(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrame(cv::cuda::Stream& stream);
        
        
        

        virtual void Init(bool firstInit);
        virtual void Serialize(ISimpleSerializer* pSerializer);
    protected:
        // Should only ever be called from the buffer thread
        virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream) = 0;
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream) = 0;

        void                                     Buffer();
        void                                     LaunchBufferThread();
        void                                     StopBufferThread();

        boost::circular_buffer<TS<SyncedMemory>> frame_buffer;
        boost::thread                            buffer_thread;
        boost::mutex                             buffer_mtx;
        int                                      playback_frame_number;
        int                                      buffer_frame_number;
    private:
        
    };

}