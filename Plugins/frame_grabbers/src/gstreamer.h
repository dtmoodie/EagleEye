#pragma once

#include "cv_capture.h"
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include "frame_grabbersExport.hpp"

namespace aq
{
    namespace Nodes
    {
        class frame_grabbers_EXPORT GrabberGstreamer: public GrabberCV
        {
        public:
            MO_DERIVE(GrabberGstreamer, GrabberCV)
                PARAM(bool, loop, true);
                MO_SIGNAL(void, eof);
            MO_END;

            bool loadData(const std::string& file_path);
            
            static int canLoad(const std::string& document);
            static void listPaths(std::vector<std::string>& paths);
        };
    }
}
