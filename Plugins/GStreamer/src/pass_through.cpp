#include "pass_through.h"
#include <EagleLib/Nodes/NodeInfo.hpp>

using namespace EagleLib;
using namespace EagleLib::Nodes;

bool h264_pass_through::ProcessImpl()
{
    if(gstreamer_string_param.modified)
    {
        create_pipeline(gstreamer_string);
        if (get_pipeline_state() != GST_STATE_PLAYING && _pipeline)
            start_pipeline();
        gstreamer_string_param.modified = false;
    }
    if(active_param.modified)
    {
        if(active)
        {
            if (get_pipeline_state() != GST_STATE_PLAYING && _pipeline)
                start_pipeline();
        }else
        {
            pause_pipeline();
        }
        active_param.modified = false;
    }
    return true;
}


MO_REGISTER_CLASS(h264_pass_through);


