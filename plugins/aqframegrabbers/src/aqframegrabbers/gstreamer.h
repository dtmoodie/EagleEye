#pragma once

#include "cv_capture.h"

namespace aqframegrabbers
{

    class aqframegrabbers_EXPORT GrabberGstreamer : public GrabberCV
    {
      public:
        static int loadTimeout() { return 10000; }
        MO_DERIVE(GrabberGstreamer, GrabberCV)
            PARAM(bool, loop, true);
            MO_SIGNAL(void, eof);
        MO_END;

        bool loadData(const std::string& file_path);

        static int canLoad(const std::string& document);
        static void listPaths(std::vector<std::string>& paths);
    };
} // namespace aqframegrabbers
