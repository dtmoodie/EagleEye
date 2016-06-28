#pragma once

#include "cv_capture.h"
#include <boost/lexical_cast.hpp>

namespace EagleLib
{
    // Frame grabber using opencv's built in openni2 bindings
    class PLUGIN_EXPORTS frame_grabber_openni2_info: public FrameGrabberInfo
    {
    public:
        frame_grabber_openni2_info();
        virtual std::string GetObjectName();
        virtual int CanLoadDocument(const std::string& document) const;
        virtual int Priority() const;
        virtual int LoadTimeout() const;
    };

    class PLUGIN_EXPORTS frame_grabber_openni2 : public frame_grabber_cv
    {
    public:
        virtual bool LoadFile(const std::string& file_path);
        virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);
        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
    protected:
        // TODO custom coordinate manager that can return points in real 3d world based on selected 2d image points
        rcc::shared_ptr<ICoordinateManager> _coordinate_manager;
    };

}