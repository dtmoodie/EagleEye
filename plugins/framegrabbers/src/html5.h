#pragma once
#ifdef HAVE_GSTREAMER
#include "aqframegrabbers_export.hpp"
#include "gstreamer.h"

namespace aq
{
    namespace nodes
    {

        class aqframegrabbers_EXPORT GrabberHTML : public GrabberGstreamer
        {
          public:
            MO_DERIVE(GrabberHTML, GrabberGstreamer)

            MO_END;
            virtual bool Load(const std::string& file_path);
            static int canLoad(const std::string& document);
        };
    }
}
#endif // HAVE_GSTREAMER
