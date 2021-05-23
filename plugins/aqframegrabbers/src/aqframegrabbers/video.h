#ifndef AQFRAMEGRABBERS_VIDEO_HPP
#define AQFRAMEGRABBERS_VIDEO_HPP
#include "cv_capture.h"

namespace aqframegrabbers
{
    class FrameGrabberVideo : public GrabberCV
    {
      public:
        MO_DERIVE(FrameGrabberVideo, GrabberCV)
        MO_END;
        static int canLoad(const std::string& document);

      protected:
    };
} // namespace aqframegrabbers

#endif // AQFRAMEGRABBERS_VIDEO_HPP
