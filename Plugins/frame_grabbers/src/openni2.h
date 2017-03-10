#pragma once

#include "cv_capture.h"
#include "frame_grabbersExport.hpp"
#include <boost/lexical_cast.hpp>

namespace aq
{
    namespace Nodes
    {
    class frame_grabbers_EXPORT frame_grabber_openni2 : public frame_grabber_cv
    {
    public:
        virtual bool LoadFile(const std::string& file_path);
        virtual TS<SyncedMemory> GetFrameImpl(int index, cv::cuda::Stream& stream);
        virtual TS<SyncedMemory> GetNextFrameImpl(cv::cuda::Stream& stream);
        virtual rcc::shared_ptr<ICoordinateManager> GetCoordinateManager();
        static int CanLoadDocument(const std::string& document);
        static int LoadTimeout();
    protected:
        // TODO custom coordinate manager that can return points in real 3d world based on selected 2d image points
        rcc::shared_ptr<ICoordinateManager> _coordinate_manager;
    };
    }
}