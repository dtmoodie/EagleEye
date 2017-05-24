#pragma once
#include <MetaObject/object/MetaObject.hpp>
#include "gstreamer.hpp"

namespace aq
{
    namespace Nodes
    {
        class PLUGIN_EXPORTS h264_pass_through: public gstreamer_sink_base
        {
        public:
            
            MO_DERIVE(h264_pass_through, gstreamer_sink_base)
                PARAM(std::string, gstreamer_string, "");
                PARAM(bool, active, false);
            MO_END;
        protected:
            bool processImpl();
            GstElement* valve = nullptr;
            bool previously_active = false;
        };
    }
}
